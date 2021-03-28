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

class RecurrentReplayBuffer():
    def __init__(self, buffer_size, sequence_length=10):
        self.buffer_size = buffer_size
        self.size = 0
        self.seq_length = sequence_length
        self.memory = []

    def update(self, prev_obs, obs, rew, act):
        self.memory.append((prev_obs, act, rew, obs))
        if self.size < self.buffer_size:
            self.size += 1
        if(len(self.memory) > self.buffer_size):
            del self.memory[0]

    def sample(self, batch_size):
        # print(self.prev_obs)
        # print(self.size)
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x-self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            #correct for sampling near beginning
            final = self.memory[max(start+1,0):end+1]
            
            #correct for sampling across episodes
            for i in range(len(final)-2, -1, -1):
                if final[i][3] is None:
                    final = final[i+1:]
                    break
                    
            #pad beginning to account for corrections
            while(len(final)<self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]))] + final
                            
            samp+=final

        #returns flattened version
        return samp, None, None

class DRQN_net(nn.Module):
    def __init__(self, in_features=4, num_actions=2, gru_size = 1):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DRQN_net, self).__init__()

        self.input_shape = in_features
        self.gru_size = gru_size
        self.fc1 = nn.Linear(in_features, 4)
        self.gru = nn.GRU(4, gru_size, num_layers=1, batch_first=True, bidirectional=False)
        # self.fc2 = nn.Linear(8, 4)
        # self.fc3 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(gru_size, num_actions)

    def forward(self, x, hx = None):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        
        x = x.view((-1,self.input_shape))
        
        #format outp for batch first gru
        feats = F.relu(self.fc1(x)).view(batch_size, sequence_length, -1)
        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(feats, hidden)
        x = self.fc2(out)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru_size, dtype=torch.float)

class DRQN:
    def __init__(self,a_space_size,q_configs,debug=False):
        self.dropped = False
        self.a_space_size = a_space_size
        self.strategy = 'drqn'
        self.q_configs = q_configs
        self.gamma = q_configs['gamma']
        self.p = q_configs['p']
        self.epsilon = q_configs['explore_config']['epsilon']
        self.decay = q_configs['explore_config']['decay_lambda']
        self.debug = debug
        self.obs_size = q_configs['obs_size']
        self.Q = DRQN_net(self.obs_size)
        self.target = DRQN_net(self.obs_size)
        self.sequence_length = q_configs['sequence_length']
        self.buffer = RecurrentReplayBuffer(q_configs['buffer_size'], self.sequence_length)
        self.position = 0
        self.batch_size = q_configs['batch_size']
        self.lr = q_configs['learning_rate']
        self.optim_eps = q_configs['optim_epsilon']
        self.optim_alpha  = q_configs['optim_alpha']
        self.target_update_freq = q_configs['target_update']
        self.optimizer = optim.Adam(self.Q.parameters(),lr=self.lr)
        self.seq = [np.zeros(self.obs_size) for j in range(self.sequence_length)]

    def pre(self,tick,prev_observation, duration):
        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(prev_observation)
            decay = self.epsilon * math.exp(-self.decay*tick)
            if random.random() < decay:
                # print("Random:",tick)
                return 0 if random.random() < self.p else 1
            # print(prev_observation)
            X = torch.tensor([self.seq], dtype=torch.float)
            row, _ = self.Q(X)
            row = row[:, -1, :]
            row = row.max(1)[1]
            # if tick > duration - 100 :
            #     print(prev_observation, row)
            # return int(row.data.max(1)[1])
            return row.item()

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.buffer.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (self.batch_size,self.sequence_length, self.obs_size)

        batch_state = torch.tensor(batch_state, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, dtype=torch.long).view(self.batch_size, self.sequence_length, -1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).view(self.batch_size, self.sequence_length)
        #get set of next states for end of each sequence
        batch_next_state = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.sequence_length)==0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), dtype=torch.uint8)
        try: #sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], dtype=torch.float).unsqueeze(dim=1)
            non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights
    
    def get_max_next_state_action(self, next_states):
        return self.target(next_states).max(dim=1)[1].view(-1, 1)

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def MSE(self, x):
        return 0.5 * x.pow(2)

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values, _ = self.Q(batch_state)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.float)
            if not empty_next_state_values:
                max_next, _ = self.target(non_final_next_states)
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = batch_reward + (self.gamma*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        
        #mask first half of losses
        split = self.sequence_length // 2
        mask = torch.zeros(self.sequence_length, dtype=torch.float)
        mask[split:] = 1.0
        mask = mask.view(1, -1)
        loss *= mask
        
        loss = loss.mean()

        return loss

    def post(self,tick,prev_observation,observation,reward,action,true_action):
        
        self.buffer.update(prev_observation, observation, reward, true_action)

        if self.buffer.size >= 4*self.batch_size and tick % self.obs_size == 0:
            # self.buffer = np.array(self.buffer)
            # batch = random.sample(self.buffer,self.batch_size)
            # batch = np.array(batch)
            batch_vars = self.prep_minibatch()

            loss = self.compute_loss(batch_vars)

            loss = loss/(self.batch_size*self.sequence_length)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.Q.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if tick % self.target_update_freq == 0:
                self.target.load_state_dict(self.Q.state_dict())
        
        if tick % 1000 == 1 and self.debug:
            print('-- tick {} --'.format(tick))
            print('prev_obs:{}, obs:{}, action:{}, reward:{}'.format(prev_observation,observation,action,reward))
            # for k in self.T:
            #     print(k,self.T[k],'action count:',self.actions_at_obs[k],'(max:{})'.format(np.argmax(self.T[k])))
            #pprint.pprint(self.T)
