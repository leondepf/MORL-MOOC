from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class EnvelopeLinearCQN(torch.nn.Module):
    '''
        Linear Controllable Q-Network, Envelope Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(EnvelopeLinearCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        ##ToDo: 1、state的shape是3维，EarlyClassification用的是Conv2d，现在用Linear，需要修改state的shape
        ##ToDo: 2、reward的shape是2维，现在用Linear，需要修改reward的shape
        ##ToDo: 3、需要调试源代码，了解Linear的输入输出是什么
        # S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 16) ##nn.Linear(4, 64)
        self.affine2 = nn.Linear((state_size + reward_size) * 16, 
                                 (state_size + reward_size) * 32) ##nn.Linear(64, 128)
        self.affine3 = nn.Linear((state_size + reward_size) * 32,
                                 (state_size + reward_size) * 64) ##nn.Linear(128, 256)
        self.affine4 = nn.Linear((state_size + reward_size) * 64,
                                 (state_size + reward_size) * 32) ##nn.Linear(256, 128)
        self.affine5 = nn.Linear((state_size + reward_size) * 32,
                                 action_size * reward_size)       ##nn.Linear(128, 8)

    def H(self, Q, w, s_num, w_num):
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1)
        w_ext = w_ext.view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def H_(self, Q, w, s_num, w_num):
        reQ = Q.view(-1, self.reward_size)

        # extend preference batch
        w_ext = w.unsqueeze(2).repeat(1, self.action_size, 1).view(-1, 2)

        # produce hte inner products
        prod = torch.bmm(reQ.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions
        prod = prod.view(-1, self.action_size)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def forward(self, state, preference, w_num=1):
        s_num = int(preference.size(0) / w_num)
        x = torch.cat((state, preference), dim=1) ## 
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        q = self.affine5(x)

        q = q.view(q.size(0), self.action_size, self.reward_size)

        hq = self.H(q.detach().view(-1, self.reward_size), preference, s_num, w_num)
        
        return hq, q


class EnvelopeCNN(nn.Module):
    def __init__(self, state_size, action_size, reward_size):
        super(EnvelopeCNN, self).__init__()
        self.state_size = state_size  ## 35
        self.action_size = action_size ## 3
        self.reward_size = reward_size ## 2

        ## TODO：可将backbone换为 Conv1d 或 RNN/LSTM/Transformer
        # self.conv1 = nn.Conv2d(in_channels=24, out_channels=128, kernel_size=3, padding=1) ##需要提前变换timestep和feature的维度
        # self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.fc = nn.Linear(128, action_size * reward_size)

        ## Conv1d
        # self.conv1 = nn.Conv1d(in_channels=24, out_channels=128, kernel_size=3, stride=1, padding=1) ##XuetangX: features+probe: 22+2
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=128, kernel_size=3, stride=1, padding=1) ##KDDCup: features+probe: 7+2
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, action_size * reward_size)

    def H(self, Q, w, s_num, w_num):
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1)
        w_ext = w_ext.view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def forward(self, x, preference, w_num=1):
        # s_num = int(preference.size(0) / w_num)

        ##TODO: x.shape = (?, 35, 22), preference.shape = (?, 2), 需要调整preference的shape
        preference = preference.unsqueeze(1) ## torch.Size([?, 1, 2])
        preference = preference.expand(-1, x.shape[1], -1) ## torch.Size([?, 35, 2])

        x = torch.cat((x, preference), dim=-1)  ## torch.Size([?, 35, 24])

        x = x.permute(0, 2, 1) ## torch.Size([?, 24, 35])
        ## TODO：操作维度需要调整, pool需要对time维度进行池化

        x = self.conv1(x) ## torch.Size([?, 128, 35])，是不是应该在特征维度上处理？
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.conv2(x) ## torch.Size([?, 256, 35])
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.conv3(x) ## torch.Size([?, 128, 35])
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        # x = F.avg_pool2d(x, kernel_size=x.size()[2:]) 
        x = F.avg_pool1d(x, kernel_size=x.size()[2])  ## torch.Size([?, 128, 1])

        # x = x.squeeze(2)
        x = x.view(-1, 128) ## torch.Size([?, 128]) 
        q = self.fc(x) ## torch.Size([?, 6])

        ##TODO: [?, 6]reshape到[?, 3, 2]后，能代表3个action的2个reward吗？
        q = q.view(q.size(0), self.action_size, self.reward_size) ## torch.Size([?, 3, 2])

        # hq = self.H(q.detach().view(-1, self.reward_size), preference, s_num, w_num)

        # return hq, q

        return 0, q
