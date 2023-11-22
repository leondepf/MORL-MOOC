from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch
import random
import os
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
parser.add_argument('--epsilon-decay', default=True, action='store_true',
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
    # monitor = Monitor(train=True, spec="-{}".format(args.method))
    # monitor.init_log(args.log, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))
    
    for num_eps in range(args.episode_num):
        ## 取出一个人的序列
        index_random_data = random.randint(0,env.x_train.shape[0]-1)
        seq = env.x_train[index_random_data]
        seq_label = env.y_train[index_random_data]
        env.reset(seq,seq_label)

        terminal = False
        loss = 0
        tot_reward = 0
        timestep = 1 #time_to_begin

        probe = None
        if args.env_name == "dst":
            probe = FloatTensor([0.8, 0.2])
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        elif args.env_name == "mooc":
            probe = FloatTensor([0.3, 0.7]) ## 可调

        while not terminal and timestep <= env.x_train.shape[1]:
            state = env.get_sequence_state(timestep) ## (35, 22)
            action = agent.act(state, timestep)
            next_timestep, reward, terminal = env.step(action) ##这个next_state返回的是当前的时间步
            # if args.log:
            #     monitor.add_log(state, action, reward, terminal, agent.w_kept)
            
            next_state = env.get_sequence_state(next_timestep)  ##如果不是等待，则next_state和state相同
            # next_state = np.reshape(next_state,(1, env.x_train.shape[1], env.x_train.shape[2], 1))
     
            agent.memorize(state, action, next_state, reward, terminal)
            # agent.memorize(state, action, next_state, reward, terminal, roi=False)
            ## TODO: 是否需要根据action的类型进行分类存储？
            ## agent.remember(state, action, reward, next_state, terminal)
            
            ## 相当于 agent.replay()
            loss += agent.learn()  ## 一次iteration

            ## TODO：这个if应该可以删掉
            if timestep > 35:
                terminal = True
                agent.reset()

            tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, timestep)
            ## 这个tot_reward是否要挪到while循环之外？
            timestep += 1
        
        # loss += agent.learn() ## 可能会有问题
        # print("loss: %0.4f" %loss)

        # ## TODO: 传入state参数是否正确？这个state是哪一个timestep的state？
        # _, q = agent.predict(state, probe)
        # if args.env_name == "dst":
        #     act_1 = q[0, 3]
        #     act_2 = q[0, 1]
        # elif args.env_name in ['ft', 'ft5', 'ft7']:
        #     act_1 = q[0, 1]
        #     act_2 = q[0, 0]
        # elif args.env_name == "mooc":
        #     ## TODO: 确定这几个Q值
        #     act_0 = q[0, 0]
        #     act_1 = q[0, 1]
        #     act_2 = q[0, 2]


        # if args.method == "crl-naive":
        #     act_1 = act_1.data.cpu()
        #     act_2 = act_2.data.cpu()
        # elif args.method == "crl-envelope":
        #     act_0 = probe.dot(act_0.data)
        #     act_1 = probe.dot(act_1.data)
        #     act_2 = probe.dot(act_2.data)
        # elif args.method == "crl-energy":
        #     act_1 = probe.dot(act_1.data)
        #     act_2 = probe.dot(act_2.data)
        
        # print("end of eps %d with total reward (1) %0.2f, the Q is %0.2f | %0.2f | %0.2f; loss: %0.4f" % (
        #     num_eps,
        #     tot_reward,
        #     act_0,
        #     act_1,
        #     act_2,
        #     # q__max,
        #     loss / timestep))

        # monitor.update(num_eps,
        #                tot_reward,
        #                act_0, 
        #                act_1,
        #                act_2,
        #                #    q__max,
        #                loss / timestep)  ## 最后一个可以换为loss

        
        ## 评估部分
        ## TODO: 改为iteration对100取余, 可以在train的过程中进行评估
        ## 这一部分是放在episode循环之内，还是放在外面也无影响？-> 无影响
        if (agent.update_count+1) % 100 == 0:  ##如果不加1，则agent.update_count为0时也会进行评估
            # acc,res,t = agent.compute_acc_batched(env,probe)
            # harmonic_mean_train = agent.harmonic_mean(acc,t)
            # print("acc_train {} ======> average_time_train {}% ======> update {}".format(acc, np.round(100.*t, 3), agent.update_number))
            acc_val,res_val,t_val = agent.compute_acc_val_batched(env,probe)
            harmonic_mean_val = agent.harmonic_mean(acc_val,t_val)
            print("iteration {} : acc_val={} , average_time_val={}%, harmonic_mean_val={} ".format((agent.update_count+1), acc_val, np.round(100.*t_val, 3), harmonic_mean_val))  
        
        # if(num_eps%20==0):
        #     print("Episode {}".format(num_eps))
        # if num_eps % 100==0 and num_eps != 0:
        #     acc,res,t = agent.compute_acc_batched(env,probe) ##相当于agent.predict()
        #     harmonic_mean_train = agent.harmonic_mean(acc,t)

        #     print("acc_train {} ======> average_time_train {}% ======> update {}".format(acc, np.round(100.*t, 3), agent.update_number))
        #     print("harmonic_mean_train {} ".format(harmonic_mean_train))

            # acc_val,res_val,t_val = agent.compute_acc_val_batched(env,probe)
            # harmonic_mean_val = agent.harmonic_mean(acc_val,t_val)
            # print("acc_val {} ======> average_time_val {}% ======> update {}".format(acc_val, np.round(100.*t_val, 3), agent.update_number))  
            # print("harmonic_mean_val {} ".format(harmonic_mean_val))
            
    
    agent.save(args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)
    # env = MOOC_Env() ##action, states, sequence, label

    # get state / action / reward sizes
    # state_size = len(env.state_spec) ## 34
    # action_size = len(env.action_spec) ## 3
    state_size = env.state_size ## 35
    action_size = env.action_size ## 3
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
