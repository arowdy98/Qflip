import numpy as np
import random
import pprint
import math
from strategies.exploration import choose_action
from strategies.estimation import estimate_value

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dtype = torch.FloatTensor

class Replay_Buffer():
    def __init__(self, buffer_size, obs_size):
        self.prev_obs = np.zeros((buffer_size, obs_size))
        self.obs = np.zeros((buffer_size, obs_size))
        self.acts = np.zeros(buffer_size)
        self.rew = np.zeros(buffer_size)
        self.buffer_size = buffer_size
        self.size = 0
        self.position = 0

    def update(self, prev_obs, obs, rew, act): 
        self.prev_obs[self.position] = prev_obs
        self.obs[self.position] = obs
        self.acts[self.position] = act
        self.rew[self.position] = rew
        self.position = (self.position + 1) % self.buffer_size

        if self.size < self.buffer_size:
            self.size += 1

    def sample(self, batch_size):
        # print(self.prev_obs)
        # print(self.size)
        idxes = random.sample(range(self.size), batch_size)
        prev_obs_batch = self.prev_obs[idxes]
        obs_batch = self.obs[idxes]
        acts_batch = self.acts[idxes]
        rew_batch = self.rew[idxes]

        return prev_obs_batch, obs_batch, acts_batch, rew_batch

class DQN(nn.Module):
    def __init__(self, in_features=4, num_actions=2):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, 4)
        # self.fc2 = nn.Linear(8, 4)
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(4, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.fc4(x)

class Q_Net2:
    def __init__(self,n_node,a_space_size,q_configs,debug=False):
        self.dropped = False
        self.a_space_size = a_space_size
        self.n_node = n_node
        self.strategy = 'q-net'
        self.q_configs = q_configs
        self.gamma = q_configs['gamma']
        self.p = q_configs['p']
        self.epsilon = q_configs['explore_config']['epsilon']
        self.decay = q_configs['explore_config']['decay_lambda']
        self.debug = debug
        self.obs_size = q_configs['obs_size']*n_node
        self.batch_size = q_configs['batch_size']
        self.lr = q_configs['learning_rate']
        self.optim_eps = q_configs['optim_epsilon']
        self.optim_alpha  = q_configs['optim_alpha']
        self.target_update_freq = q_configs['target_update']
        self.Q = []
        self.target = []
        self.buffer = []
        self.optim = []
        for i in range(n_node):
            self.Q.append(DQN(self.obs_size))
            self.target.append(DQN(self.obs_size))
            self.buffer.append(Replay_Buffer(q_configs['buffer_size'], self.obs_size))
            self.optim.append(optim.Adam(self.Q[i].parameters(),lr=self.lr))

    def pre(self,tick,prev_observation, duration):
        decay = self.epsilon * math.exp(-self.decay*tick)
        res = []
        # print(self.n_node)
        prev_observation = torch.from_numpy(np.array(prev_observation)).type(dtype).unsqueeze(0)
        for i in range(self.n_node):
            if random.random() < decay:
                # print("Random:",tick)
                res.append(0) if random.random() < self.p else res.append(1)
                continue
            # print(prev_observation) 
            row = self.Q[i](prev_observation)
            # if tick > duration - 100 :
            # print(prev_observation, row)
            res.append(int(row.data.max(1)[1]))
            # print(res)
        return res

    def post(self,tick,prev_observation,observation,reward,action,true_action):
        
        for i in range(self.n_node):
            self.buffer[i].update(prev_observation, observation, reward, true_action[i])

            if self.buffer[i].size >= 4*self.batch_size and tick % 4 == 0:
                # self.buffer = np.array(self.buffer)
                # batch = random.sample(self.buffer,self.batch_size)
                # batch = np.array(batch)
                prev_obs_batch, obs_batch, act_batch, rew_batch = self.buffer[i].sample(self.batch_size)
                
                prev_obs_batch = torch.from_numpy(prev_obs_batch).type(dtype)
                obs_batch = torch.from_numpy(obs_batch).type(dtype)
                rew_batch = torch.from_numpy(rew_batch).type(dtype)
                act_batch = torch.from_numpy(act_batch).long()

                current_Q = self.Q[i](prev_obs_batch).gather(1, act_batch.unsqueeze(1))
                next_Q = self.target[i](obs_batch).detach().max(1)[0]
                target_Q = rew_batch + (self.gamma * next_Q)

                loss = F.smooth_l1_loss(current_Q, target_Q.unsqueeze(1))
                self.optim[i].zero_grad()
                loss.backward()

                for param in self.Q[i].parameters():
                    param.grad.data.clamp_(-1, 1)
                
                self.optim[i].step()

                if tick % self.target_update_freq == 0:
                    self.target[i].load_state_dict(self.Q[i].state_dict())
        
        if tick % 1000 == 1 and self.debug:
            print('-- tick {} --'.format(tick))
            print('prev_obs:{}, obs:{}, action:{}, reward:{}'.format(prev_observation,observation,action,reward))
            # for k in self.T:
            #     print(k,self.T[k],'action count:',self.actions_at_obs[k],'(max:{})'.format(np.argmax(self.T[k])))
            #pprint.pprint(self.T)
