from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, model, args, is_train=False):
        self.model = copy.deepcopy(model)
        self.model_ = model
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num

        self.beta            = args.beta
        self.beta_init       = args.beta
        self.homotopy        = args.homotopy  ## False
        self.beta_uplim      = 1.00
        self.tau             = 1000.
        self.beta_expbase    = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./args.episode_num))
        self.beta_delta      = self.beta_expbase / self.tau

        self.exploration_rate   = 0.5
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995

        self.trans_mem = deque()
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        self.priority_mem = deque()

        self.memory_0 = deque(maxlen=3000)
        self.memory_1 = deque(maxlen=3000)
        self.memory_2 = deque(maxlen=3000)

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

        # self.state_size        = 30 ## KDD2015 Dataset
        self.state_size        = 35 ## XuetangX Dataset
        self.action_size       = 3


    def act(self, state, index, preference=None):
        
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)  
        
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.model_.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / \
                            torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept
        state = torch.from_numpy(state).type(FloatTensor)

        ##TODO：检查这里的输入输出shape
        ##这里调用类EnvelopeCNN的forward函数，输入state和preference，输出Q值
        _, Q = self.model_(
            Variable(state.unsqueeze(0)), ##state.unsqueeze(0)的shape是[1, 35, 22]
            Variable(preference.unsqueeze(0))) ##preference.unsqueeze(0)的shape是[1, 2]
        ## Q: torch.Size([1, 3, 2])

        Q = Q.view(-1, self.model_.reward_size) ## torch.Size([3, 2])

        Q = torch.mv(Q.data, preference) ## torch.Size([3])

        ## TODO
        if np.random.rand() <= self.exploration_rate and self.is_train:
            action = random.randrange(self.action_size) 
        else:
            action = Q.max(0)[1].cpu().numpy() ##找到具有最大值的索引。max函数返回两个值：最大值和最大值的索引。只使用索引，该索引代表具有最高Q值的动作。
            action = int(action)

        return action

    
        # if np.random.rand() <= self.exploration_rate:
        #     action = random.randrange(self.action_size - 1) + 1  

        # if self.is_train and (len(self.trans_mem) < self.batch_size or \
        #                     torch.rand(1)[0] < self.epsilon):
        # action = np.random.choice(self.model_.action_size, 1)[0]
        # action = int(action)

        # return action

    def memorize(self, state, action, next_state, reward, terminal):
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  #state (35, 22)
            action,  #action (1,)
            torch.from_numpy(next_state).type(FloatTensor),  # next state (35, 22)
            torch.from_numpy(reward).type(FloatTensor),  # reward (2,)
            terminal))  # terminal True/False

        ##TODO:从这里开始检查输入输出的shape
        # randomly produce a preference for calculating priority
        # preference = self.w_kept
        preference = torch.randn(self.model_.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
        state = torch.from_numpy(state).type(FloatTensor)

        _, q = self.model_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False)) ## torch.Size([1, 3, 2])
        ## 将preference加入state预测得到的Q值

        q = q[0, action].data ## torch.Size([2]),tensor([-0.1004,  0.0867], device='cuda:0')
        wq = preference.dot(q) ##（1,），对预测的Q值进行加权求和

        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor)) ##（1,），对真实的reward进行加权求和
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            _, hq = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False)) 
            ## 计算 next_state 的 Q_target
            
            ## TODO： AttributeError: 'int' object has no attribute 'data'，应该是跟next_state的shape有关
            hq = hq.data[0,action]
            whq = preference.dot(hq) ## 对预测的 Q_target 进行加权求和
            p = abs(wr + self.gamma * whq - wq) ## Eq(6), Q_target - Q_predict
        else:
            print(self.beta)
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            if self.homotopy:
                self.beta += self.beta_delta
                self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
            p = abs(wr - wq)  ## TDError
        p += 1e-5

        self.priority_mem.append(
            p
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def remember(self, state, action, reward, next_state, done):
        if action == 0:  
            self.memory_0.append((state, action, reward, next_state, done))
        if action == 1:
            self.memory_1.append((state, action, reward, next_state, done))
        if action == 2 :
            self.memory_2.append((state, action, reward, next_state, done))

    def sample(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self):
        ## TODO：如果train.py中使用agent.remember()函数，则加入以下代码：
        '''
        if (len(self.memory_0) < (sample_batch_size/3) or len(self.memory_1) < (sample_batch_size/3) or len(self.memory_2) < (sample_batch_size /3)):
            return ## 如果三个memory中有一个小于sample_batch_size/3，就不进行训练，不会执行后面的语句
        
        ## 当3个memory存满后，才会执行下面的语句，这样处理保证了0/1/2三个动作的采样是一致的，不会导致不均衡问题
        sample_size = int(sample_batch_size/3)
        sample_batch_0 = random.sample(self.memory_0, sample_size)
        sample_batch_1 = random.sample(self.memory_1, sample_size + 1)
        sample_batch_2 = random.sample(self.memory_2, sample_size + 1)
        
        sample_batch_1.extend(sample_batch_2)
        sample_batch_0.extend(sample_batch_1)
        random.shuffle(sample_batch_0)
        sample_batch = sample_batch_0
        '''

        if len(self.trans_mem) > self.batch_size:

            self.update_count += 1

            action_size = self.model_.action_size
            reward_size = self.model_.reward_size

            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)  ##优先采样
            batchify = lambda x: list(x) * self.weight_num
            state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))

            w_batch = np.random.randn(self.weight_num, reward_size)
            w_batch = np.abs(w_batch) / np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
            w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)

            __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                Variable(w_batch), w_num=self.weight_num)

            # detach since we don't want gradients to propagate
            # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
            # 					  Variable(w_batch, volatile=True), w_num=self.weight_num)
            _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                               Variable(w_batch, requires_grad=False))
            w_ext = w_batch.unsqueeze(2).repeat(1, action_size, 1)
            w_ext = w_ext.view(-1, self.model.reward_size)
            _, tmpQ = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                  Variable(w_batch, requires_grad=False))

            tmpQ = tmpQ.view(-1, reward_size)
            # print(torch.bmm(w_ext.unsqueeze(1),
            # 			    tmpQ.data.unsqueeze(2)).view(-1, action_size))
            act = torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
                            tmpQ.unsqueeze(2)).view(-1, action_size).max(1)[1]

            HQ = DQ.gather(1, act.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()

            nontmlmask = self.nontmlinds(terminal_batch)
            with torch.no_grad():
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num,
                                             reward_size).type(FloatTensor))
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                # Tau_Q.volatile = False
                Tau_Q += Variable(torch.cat(reward_batch, dim=0))

            actions = Variable(torch.cat(action_batch, dim=0))

            Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
                         ).view(-1, reward_size)
            Tau_Q = Tau_Q.view(-1, reward_size)

            wQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                           Q.unsqueeze(2)).squeeze()

            wTQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                            Tau_Q.unsqueeze(2)).squeeze()

            # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
            loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
            loss += (1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model_.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay

            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            return loss.data

        return 0.0

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
        if self.homotopy:
            self.beta += self.beta_delta
            self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

    def predict(self, state, probe):
        ## TODO: 'numpy.ndarray' object has no attribute 'unsqueeze'
        return self.model(Variable(FloatTensor(state).unsqueeze(0), requires_grad=False),
                          Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name):
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))


    def find_preference(
            self,
            w_batch,
            target_batch,
            pref_param):

        with torch.no_grad():
            w_batch = FloatTensor(w_batch)
            target_batch = FloatTensor(target_batch)

        # compute loss
        pref_param = FloatTensor(pref_param)
        pref_param.requires_grad = True
        sigmas = FloatTensor([0.001]*len(pref_param))
        dist = torch.distributions.normal.Normal(pref_param, sigmas)
        pref_loss = dist.log_prob(w_batch).sum(dim=1) * target_batch

        self.optimizer.zero_grad()
        # Total loss
        loss = pref_loss.mean()
        loss.backward()
        
        eta = 1e-3
        pref_param = pref_param + eta * pref_param.grad
        pref_param = simplex_proj(pref_param.detach().cpu().numpy())
        # print("update prefreence parameters to", pref_param)

        return pref_param
    
    


# projection to simplex
def simplex_proj(x):
    y = -np.sort(-x)
    sum = 0
    ind = []
    for j in range(len(x)):
        sum = sum + y[j]
        if y[j] + (1 - sum) / (j + 1) > 0:
            ind.append(j)
        else:
            ind.append(0)
    rho = np.argmax(ind)
    delta = (1 - (y[:rho+1]).sum())/(rho+1)
    return np.clip(x + delta, 0, 1)
