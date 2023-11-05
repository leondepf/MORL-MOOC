from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch
import random
from utils.monitor import Monitor
from envs.mo_env import MultiObjectiveEnv


parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
                    help='environment to train on: dst | ft | ft5 | ft7 | mooc')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# TRAINING
parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=2000, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='(initial) beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')
# LOG & SAVING
parser.add_argument('--serialize', default=False, action='store_true',
                    help='serialize a model')
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='path for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
parser.add_argument('--log', default='crl/naive/logs/', metavar='LOG',
                    help='path for recording training informtion')

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def train(env, agent, args):
    monitor = Monitor(train=True, spec="-{}".format(args.method))
    monitor.init_log(args.log, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))
    
    for num_eps in range(args.episode_num):
        index_random_data = random.randint(0,env.x_train.shape[0]-1)
        seq = env.x_train[index_random_data]
        seq_label = env.y_train[index_random_data]
        env.reset(seq,seq_label)

        terminal = False
        loss = 0
        cnt = 0
        tot_reward = 0
        timestep = 1 #time_to_begin

        probe = None
        if args.env_name == "dst":
            probe = FloatTensor([0.8, 0.2])
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        elif args.env_name == "mooc":
            probe = FloatTensor([0.3, 0.7])

        while not terminal and timestep <= env.x_train.shape[1]:
            state = env.observe() ## (35, 22)
            action = agent.act(state, timestep)
            next_timestep, reward, terminal = env.step(action) ##这个next_state返回的是当前的时间步
            if args.log:
                monitor.add_log(state, action, reward, terminal, agent.w_kept)
            
            next_state = env.get_sequence_state()  ##
            # next_state = np.reshape(next_state,(1, env.x_train.shape[1], env.x_train.shape[2], 1))
     
            agent.memorize(state, action, next_state, reward, terminal)
            ## TODO: 是否需要根据action的类型进行分类存储？
            ## agent.remember(state, action, reward, next_state, terminal)
            
            # state = next_state
            loss += agent.learn()

            ## TODO：这个if应该可以删掉
            if cnt > 35:
                terminal = True
                agent.reset()

            tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            cnt = cnt + 1
            timestep += 1

        ## TODO: 评估部分
        '''
        if(index_episode%20==0):
            print("Episode {}".format(index_episode))
        if index_episode % 100==0 and index_episode != 0:
            acc,res,t = self.agent.compute_acc_batched()
            harmonic_mean_train = self.agent.harmonic_mean(acc,t)

            acc_val,res_val,t_val = self.agent.compute_acc_val_batched()
            harmonic_mean_val = self.agent.harmonic_mean(acc_val,t_val)

            # acc,res,t = self.agent.compute_acc()
            # acc_val,res_val,t_val = self.agent.compute_acc_val()
            # print("acc_train {} ======> average_time_train {} ======> update {}".format(acc, np.mean(t), self.agent.update_number))
            # print("acc_val {} ======> average_time_val {} ======> update {}".format(acc_val, np.mean(t_val), self.agent.update_number))
            print("acc_train {} ======> average_time_train {}% ======> update {}".format(acc, np.round(100.*t, 3), self.agent.update_number))
            print("harmonic_mean_train {} ".format(harmonic_mean_train))
            print("acc_val {} ======> average_time_val {}% ======> update {}".format(acc_val, np.round(100.*t_val, 3), self.agent.update_number))  
            print("harmonic_mean_val {} ".format(harmonic_mean_val))
            if acc > 0.9 :
                self.agent.save_weight()
        '''

        ## TODO: 修改predict函数的参数
        _, q = agent.predict(state, probe)
        if args.env_name == "dst":
            act_1 = q[0, 3]
            act_2 = q[0, 1]
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            act_1 = q[0, 1]
            act_2 = q[0, 0]
        elif args.env_name == "mooc":
            ## TODO: 确定这几个Q值
            act_0 = q[0, 0]
            act_1 = q[0, 1]
            act_2 = q[0, 2]


        if args.method == "crl-naive":
            act_1 = act_1.data.cpu()
            act_2 = act_2.data.cpu()
        elif args.method == "crl-envelope":
            act_0 = probe.dot(act_0.data)
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        elif args.method == "crl-energy":
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        
        print("end of eps %d with total reward (1) %0.2f, the Q is %0.2f | %0.2f | %0.2f; loss: %0.4f" % (
            num_eps,
            tot_reward,
            act_0,
            act_1,
            act_2,
            # q__max,
            loss / cnt))
        monitor.update(num_eps,
                       tot_reward,
                       act_1,
                       act_2,
                       #    q__max,
                       loss / cnt)
    
    agent.save(args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))


if __name__ == '__main__':
    args = parser.parse_args()

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)
    # env = MOOC_Env() ##action, states, sequence, label

    # get state / action / reward sizes
    state_size = len(env.state_spec) ## 34
    action_size = len(env.action_spec) ## 3
    reward_size = len(env.reward_spec) ## 2

    # generate an agent for initial training
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
        from crl.naive.models import get_new_model
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
        from crl.envelope.models import get_new_model
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
        from crl.energy.models import get_new_model

    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    else:
        model = get_new_model(args.model, state_size, action_size, reward_size)
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
