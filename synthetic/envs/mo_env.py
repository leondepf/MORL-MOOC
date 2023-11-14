from __future__ import absolute_import, division, print_function
import numpy as np
from .deep_sea_treasure import DeepSeaTreasure
from .fruit_tree import FruitTree
from .mooc import MOOC_Env


class MultiObjectiveEnv(object):

    def __init__(self, env_name="deep_sea_treasure"):
        if env_name == "dst":
            self.env = DeepSeaTreasure()
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec
        if env_name == "ft":
            self.env = FruitTree()
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec
        if env_name == "ft5":
            self.env = FruitTree(5)
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec
        if env_name == "ft7":
            self.env = FruitTree(7)
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec
        if env_name == "mooc":
            ## KDD2015 Dataset
            # self.x_train, self.x_test, self.y_train, self.y_test = np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/resampled_X_train_all_timesteps.npy"),\
            # np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/X_test_all_timesteps.npy"),\
            # np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/resampled_C_train_all_timesteps.npy"),\
            # np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/C_test_all_timesteps.npy")

            ## XuetaingX Dataset
            self.x_train, self.x_test, self.y_train, self.y_test = np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/resampled_X_train_all_timesteps.npy"),\
            np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/X_test_all_timesteps.npy"),\
            np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/resampled_C_train_all_timesteps.npy"),\
            np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/C_test_all_timesteps.npy")
            ## (15902, 35, 22) (4768, 35, 22) (15902,) (4768,)


            # self.state_size        = 30 ## KDD2015 Dataset
            self.state_size  = 35 ## XuetangX Dataset
            self.action_size = 3

            #spec表示取值范围
            self.action_spec = [0,1,2]
            self.state_spec = [j for j in range(1,self.state_size)] ## 1~34, 实际为timestep

            self.env = MOOC_Env(self.action_spec, self.state_spec, self.x_train[0], self.y_train[0]) ##action, states, sequence, label
            
            self.reward_spec = self.env.reward

    def reset(self, sequence, label):
        ''' reset the enviroment '''
        self.env.reset(sequence, label)

    # def observe(self):
    #     ''' reset the enviroment '''
    #     return self.env.get_sequence_state()

    def step(self, action):
        ''' process one step transition (s, a) -> s'
            return (s', r, terminal)
        '''
        return self.env.step(action)

    def get_sequence_state(self, timestep):
        return self.env.get_sequence_state(timestep)


if __name__ == "__main__":
    '''
        Test ENVs
    '''
    dst_env = MultiObjectiveEnv("mooc")
    dst_env.reset()
    terminal = False
    print("DST STATE SPEC:", dst_env.state_spec)
    print("DST ACTION SPEC:", dst_env.action_spec)
    print("DST REWARD SPEC:", dst_env.reward_spec)
    while not terminal:
        state = dst_env.observe()
        action = np.random.choice(2, 1)[0]
        next_state, reward, terminal = dst_env.step(action)
        print("s:", state, "\ta:", action, "\ts':", next_state, "\tr:", reward)
    print("AN EPISODE ENDS")
