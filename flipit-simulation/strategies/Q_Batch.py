import numpy as np
import random
import pprint
from strategies.exploration import choose_action
from strategies.estimation import estimate_value


class Q_Batch:
    def __init__(self,a_space_size,q_configs,debug=False):
        self.dropped = False
        self.a_space_size = a_space_size
        self.strategy = 'q-batch'
        self.q_configs = q_configs
        self.gamma = q_configs['gamma']
        self.p = q_configs['p']
        self.exploration_type = q_configs['exploration_type']
        self.estimate_type = q_configs['estimate_type']
        self.debug = debug
        self.T = {}
        self.actions_at_obs = {}
        self.buffer = []
        self.batch_size = q_configs['batch_size']

    def pre(self,tick,prev_observation):
        if not prev_observation in self.T or self.T[prev_observation][0] == self.T[prev_observation][1]:
                if random.random() < self.p:
                    action = 0
                else:
                    action = 1
        elif self.exploration_type == 'noise':
            action = choose_action.noise(self.q_configs['explore_config']['sigma'],self.T[prev_observation])
        elif self.exploration_type == 'epsilon-greedy':
            action = choose_action.epsilon_greedy(self.q_configs['explore_config']['epsilon'],self.T[prev_observation],tick,self.q_configs['explore_config']['decay_lambda'])
        elif self.exploration_type == 'epsilon-greedy-visit-decay':
            action = choose_action.epsilon_greedy_visit_decay(self.q_configs['explore_config']['epsilon'],self.T[prev_observation],sum(self.actions_at_obs[prev_observation]),self.q_configs['explore_config']['decay_lambda'],self.p)
        elif self.exploration_type == 'uniform-epsilon-greedy-visit-decay':
            action = choose_action.uniform_epsilon_greedy_visit_decay(self.q_configs['explore_config']['epsilon'],self.T[prev_observation],sum(self.actions_at_obs[prev_observation]),self.q_configs['explore_config']['decay_lambda'])
        else:
            raise NotImplementedError


        return action

    def post(self,tick,prev_observation,observation,reward,action,true_action):
        if prev_observation not in self.actions_at_obs:
            self.actions_at_obs[prev_observation] = [0,0]
        self.actions_at_obs[prev_observation][true_action] += 1

        self.buffer.append([prev_observation,observation,reward,true_action])

        if len(self.buffer) == self.batch_size:
            temp = self.T
            for event in self.buffer:
                self.alpha = 1/self.actions_at_obs[event[0]][event[3]]

                if not event[0] in temp:
                    temp[event[0]] = [0,0]
                future_val = np.max(self.T[event[1]]) if event[1] in self.T else 0
                current_estimate = temp[event[0]][event[3]]
                
                if self.estimate_type == 'loss':
                    rew = estimate_value.loss(current_estimate,future_val,self.gamma,self.alpha,event[2])
                elif self.estimate_type == 'future':
                    rew = estimate_value.future(current_estimate,future_val,self.gamma,self.alpha,event[2])
                elif self.estimate_type == 'td':
                    rew = estimate_value.td(current_estimate,future_val,self.gamma,self.alpha,event[2])
                else:
                    raise NotImplementedError
                
                temp[event[0]][event[3]] = round(rew,10)
            self.T = temp
            temp = None
            self.buffer = []

        if tick % 1000 == 1 and self.debug:
            print('-- tick {} --'.format(tick))
            print('prev_obs:{}, obs:{}, action:{}, reward:{}'.format(prev_observation,observation,action,reward))
            for k in self.T:
                print(k,self.T[k],'action count:',self.actions_at_obs[k],'(max:{})'.format(np.argmax(self.T[k])))
            #pprint.pprint(self.T)
