import numpy as np
import random
import pprint
import math
from strategies.exploration import choose_action
from strategies.estimation import estimate_value

class Greedy_Max():
    def __init__(self,n_node,q_configs):
        self.n_node = n_node
        self.strategy = 'greedy-max'
        self.lr = q_configs['lr_greedy']
        self.beta = [0.0]*self.n_node
        self.position = 0
        self.max_r = 0
        self.prev_max = 0
        self.max_pos = 0
        self.curr = 0
        self.flag = 0

    def pre(self,tick,prev_observation, duration):
        res = []
        for i in range(self.n_node):
            if i == self.position:
                if self.beta[i] + self.lr > random.random():
                    res.append(1)
                else:
                    res.append(0)
            else:
                if self.beta[i] > random.random():
                    res.append(1)
                else:
                    res.append(0)
        return res

    def post(self,tick,prev_observation,observation,reward,action,true_action):
        if(tick>0 and tick%100 == 0):
            if(self.curr-self.prev_max >self.max_r-self.prev_max):
                self.max_r = self.curr
                self.max_pos = self.position
                self.flag = 1
            self.position = int((tick/100)%self.n_node)
            if(self.position == 0 and self.flag == 1):
                self.beta[self.max_pos] += self.lr
                self.prev_max = self.max_r
                self.flag = 0
            self.curr = 0

        self.curr += reward
        
        if tick % 1000 == 1:
            print('-- tick {} --'.format(tick))
            print('prev_obs:{}, obs:{}, action:{}, reward:{}'.format(prev_observation,observation,action,reward))
            print(self.beta)
            # for k in self.T:
            #     print(k,self.T[k],'action count:',self.actions_at_obs[k],'(max:{})'.format(np.argmax(self.T[k])))
            #pprint.pprint(self.T)
