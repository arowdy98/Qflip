import numpy as np
import random
import pprint
import math
from strategies.exploration import choose_action
from strategies.estimation import estimate_value
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

dtype = torch.FloatTensor
eps = np.finfo(np.float32).eps.item()
normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([0.05]))

class Policy(nn.Module):
    def __init__(self, in_features=4, num_actions=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(in_features, 4)
        # self.fc2 = nn.Linear(8, 4)
        # self.fc3 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(4, num_actions)

        self.saved_log_probs = []
        self.rewards = []
        self.add_noise()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return F.softmax(self.fc2(x), dim=1)

    def add_noise(self):
        t1 = normal_dist.sample((self.fc1.weight.view(-1).size())).reshape(self.fc1.weight.size())
        t2 = normal_dist.sample((self.fc2.weight.view(-1).size())).reshape(self.fc2.weight.size())
        with torch.no_grad():
            self.fc1.weight.add_(t1)
            self.fc2.weight.add_(t2)
        print("Noise added to policy parameters.")

class Policy_Net:
    def __init__(self,a_space_size,q_configs,debug=False):
        self.dropped = False
        self.a_space_size = a_space_size
        self.strategy = 'policy-net'
        self.q_configs = q_configs
        self.gamma = q_configs['gamma']
        self.p = q_configs['p']
        self.epsilon = q_configs['explore_config']['epsilon']
        self.decay = q_configs['explore_config']['decay_lambda']
        self.debug = debug
        self.obs_size = q_configs['obs_size']
        self.Policy = Policy(self.obs_size)
        self.position = 0
        self.batch_size = q_configs['batch_size']
        self.lr = q_configs['learning_rate']
        self.optim_eps = q_configs['optim_epsilon']
        self.optim_alpha  = q_configs['optim_alpha']
        self.target_update_freq = q_configs['target_update']
        self.optim = optim.Adam(self.Policy.parameters(),lr=self.lr)

    def pre(self, tick, state, duration):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.Policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.Policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def post(self,tick,prev_observation,observation,reward,action,true_action):
        
        self.Policy.rewards.append(reward)

        if tick%100 == 99:
            R = 0
            policy_loss = []
            returns = []
            for r in self.Policy.rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            self.optim.zero_grad()

            for log_prob, R in zip(self.Policy.saved_log_probs, returns):
                # policy_loss.append(-log_prob * R)
                loss = -log_prob * R
                loss.backward()
        
            # policy_loss = torch.cat(policy_loss).sum()
            # policy_loss.backward()

            # for param in self.Policy.parameters():
            #     param.grad.data.clamp_(-1, 1)
            
            self.optim.step()

            decay = self.epsilon * math.exp(-self.decay*tick)
            if random.random() < decay:
                # print("Random:",tick)
                self.Policy.add_noise()
            
            del self.Policy.rewards[:]
            del self.Policy.saved_log_probs[:]
        
        if tick % 1000 == 1 and self.debug:
            print('-- tick {} --'.format(tick))
            print('prev_obs:{}, obs:{}, action:{}, reward:{}'.format(prev_observation,observation,action,reward))
            # for k in self.T:
            #     print(k,self.T[k],'action count:',self.actions_at_obs[k],'(max:{})'.format(np.argmax(self.T[k])))
            #pprint.pprint(self.T)
