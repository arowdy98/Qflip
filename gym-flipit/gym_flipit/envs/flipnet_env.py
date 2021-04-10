import gym
from gym import error, utils, spaces
from gym.utils import seeding
import random
from gym_flipit.envs.flip_node import FlipNode
from gym_flipit.envs.strategies import periodic, exponential, uniform, normal, custom
from gym_flipit.envs.state import reset_state,set_obs_space,set_state
from gym_flipit.envs.rew import calc_rew

class FlipNetEnv(gym.Env):
    #p0 Defender
    #p1 Attacker
    def __init__(self,state_type='opp_LM',rew_type='constant_minus_cost_norm',rew_configs={'val':10,'c':'5'},p0='binomial',p0_configs={'n_node':1,'n_seed':1,'edges':[[1,2]]},duration=1000,p0_move_cost=1,p1_move_cost=5):
        # self.config(state_type,rew_type,rew_configs,p0,p0_configs,duration,p0_move_cost,p1_move_cost)
        pass

    def config(self,state_type,rew_type,rew_configs,p0,p0_configs,duration=100,p0_move_cost=1,p1_move_cost=5):
        self.duration = duration
        self.state_type = state_type
        self.n_node = p0_configs['n_node']
        self.set_obs_space()
        self.action_space = spaces.Tuple([spaces.Discrete(2)]*p0_configs['n_node'])
        self.rew_configs = rew_configs
        self.rew_type = rew_type
        self.p0_configs = p0_configs
        self.n_seed = p0_configs['n_seed']
        self.edges = p0_configs['edges']
        # print(self.edges)
        self.nodes = []
        self.adj = []
        self.node_controller = [1]*self.n_seed + [0]*self.n_node
        for i in range(self.n_node+self.n_seed):
            self.adj.append([0]*(self.n_node+self.n_seed))

        for i,j in self.edges:
            self.adj[i-1][j-1] = 1
        # print(p0_configs)
        for i in range(self.n_node):
            # print(p0_configs['beta'])
            p0_configs_indi = p0_configs.copy()
            p0_configs_indi['delta'] = p0_configs['delta'][i]
            # p0_configs_indi['beta'] = p0_configs['beta'][i]
            # p0_configs_indi['mean'] = p0_configs['mean'][i]
            # p0_configs_indi['std_dev'] = p0_configs['std_dev'][i]
            self.nodes.append(FlipNode(state_type,rew_type,rew_configs,p0,p0_configs_indi,duration,p0_move_cost,p1_move_cost))
        self.reset()
    
    def reset(self):
        self.player_total_gain = [0,0]
        self.player_total_move_cost = [0,0]
        self.tick = 0
        self.reset_state()
        self.found_FM = False
        return self.state

    def step(self, action):
        self.tick += 1
        # if both players play, defender gets control
        node_copy = self.node_controller.copy()
        rew = 0
        rew_list = []
        self.player_total_move_cost = [0,0]
        tru_act = []
        for i in range(self.n_node):
            if action[i] == 1 and node_copy[self.n_seed+i] == 1:
                observation, reward, done, info = self.nodes[i].step(1)
            elif action[i] == 1 and node_copy[self.n_seed+i] == 0:
                for j in range(self.n_node+self.n_seed):
                    if self.adj[j][self.n_seed+i] == 1 and node_copy[j] == 1:
                        observation, reward, done, info = self.nodes[i].step(1)
                        break
                else:
                    observation, reward, done, info = self.nodes[i].step(0)
            elif action[i] == 0:
                observation, reward, done, info = self.nodes[i].step(0)
            tru_act.append(info['true_action'])
            self.node_controller[self.n_seed+i] = self.nodes[i].controller
            self.state[i] = observation
            rew += reward
            rew_list.append(reward)
            self.player_total_gain[self.nodes[i].controller] += 1
            self.player_total_move_cost[0] += self.nodes[i].player_total_move_cost[0]
            self.player_total_move_cost[1] += self.nodes[i].player_total_move_cost[1]
        return self.state, rew/float(self.n_node), done, {'true_action':tru_act}
        # return self.state, rew_list, done, {'true_action':tru_act}
    
    def set_obs_space(self):
        if self.state_type == 'opp_LM':
            self.observation_space = spaces.Tuple([set_obs_space.opp_LM(self.duration)]*self.n_node)
        elif self.state_type == 'own_LM':
            self.observation_space = spaces.Tuple([set_obs_space.own_LM(self.duration)]*self.n_node)
        elif self.state_type == 'composite':
            self.observation_space = spaces.Tuple([set_obs_space.composite(self.duration)]*self.n_node)
        else:
            raise NotImplementedError

    def reset_state(self):
        if self.state_type == 'opp_LM':
            self.state = [reset_state.opp_LM()]*self.n_node
        elif self.state_type == 'own_LM':
            self.state = [reset_state.own_LM()]*self.n_node
        elif self.state_type == 'composite':
            self.state = [reset_state.composite()]*self.n_node
        else:
            raise NotImplementedError

    '''
    Calculate benefit
    '''
    def calc_benefit(self, player):
        return self.player_total_gain[player] - self.player_total_move_cost[player]

    def calc_avg_benefit(self, player):
        return self.calc_benefit(player, self.tick) / self.tick
