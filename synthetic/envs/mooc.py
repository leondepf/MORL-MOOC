from __future__ import absolute_import, division, print_function
from operator import index
import os
import numpy as np
from numpy import array
import random
from collections import deque



def get_padding_sequence(sequence, t):
    size = sequence.shape[0]
    seq = sequence[:t]
    seq = np.append(seq, np.zeros((size-t,sequence.shape[1])),axis=0)
    return seq

def get_padding_sequence_batched(sequence, t):
    size = sequence.shape[1]
    seq = sequence[:,:t]
    seq = np.append(seq, np.zeros((sequence.shape[0],size-t,sequence.shape[2])),axis=1)
    return seq



class MOOC_Env():
    def __init__(self, action, states, sequence, label):
        self.action = action
        self.states = states
        self.sequence = {"data" : sequence, "label" : label}
        self.timestep = 1 ##当前时间步,并不是强化学习中的状态
        self.done = False
        ## TODO: 这个reward需要调整
        self.reward = [[0, 14], [-1, 0]]
        self.param_lambda = 0.001
        self.param_p = 1/3

        
    def step(self, action):
        if action == 0:  
            if not self.done and self.timestep < len(self.sequence["data"]):
                ## TODO: 将这个reward修改为arrary形式
                acc_reward = 0
                time_penalty = -(self.param_lambda * (self.timestep ** self.param_p))
                self.reward = np.array([acc_reward, time_penalty])
                self.timestep = self.timestep + 1 ##时间步加1，即进入下一个时间步
                
        if action == 1:  
            if not self.done and self.timestep <= len(self.sequence["data"]):
                # if self.sequence["label"] == 1:
                if self.sequence["label"]+1 == 1:
                    acc_reward = 1
                    time_penalty = 1.0 / self.timestep
                    self.reward = np.array([acc_reward, time_penalty])
                else:
                    acc_reward = -1
                    time_penalty = 1.0 / self.timestep
                    self.reward = np.array([acc_reward, time_penalty])
            
            self.done = True
   
        if action == 2:  
            if not self.done and self.timestep <= len(self.sequence["data"]):
                # if self.sequence["label"] == 2:
                if self.sequence["label"]+1 == 2:
                    acc_reward = 1
                    time_penalty = 1.0 / self.timestep
                    self.reward = np.array([acc_reward, time_penalty])
                else:
                    acc_reward = -1
                    time_penalty = 1.0 / self.timestep
                    self.reward = np.array([acc_reward, time_penalty])
            self.done = True
     
        return self.timestep, self.reward, self.done
            
            
    def reset(self, sequence, label):
        self.sequence = {"data" : sequence, "label" : label}
        self.timestep = 1
        self.done = False      
        return self.timestep, self.done  
    
    def set_label(self,label):
        self.sequence["label"] = label
            
    def get_sequence_state(self):
        return get_padding_sequence(self.sequence["data"], self.timestep) 
    
    def get_initial_sequence(self):
        return self.sequence["data"]
    
    def get_state(self):
        return self.timestep
            
            
    def render(self, mode='human'):
        return 0
    def close (self):
        return 0