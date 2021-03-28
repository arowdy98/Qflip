import random
from gym_flipit.envs.strategies import periodic, exponential, uniform, normal, custom, binomial
from gym_flipit.envs.state import reset_state,set_obs_space,set_state
from gym_flipit.envs.rew import calc_rew

class FlipNode():

    def __init__(self,state_type='opp_LM',rew_type='constant_minus_cost_norm',rew_configs={'val':10,'c':'5'},p0='periodic',p0_configs={'delta':10},duration=1000,p0_move_cost=1,p1_move_cost=5):
        self.config(state_type,rew_type,rew_configs,p0,p0_configs,duration,p0_move_cost,p1_move_cost)

    def config(self,state_type,rew_type,rew_configs,p0,p0_configs,duration=100,p0_move_cost=1,p1_move_cost=5):
        self.duration = duration
        self.state_type = state_type
        self.rew_configs = rew_configs
        self.rew_type = rew_type
        self.p0_configs = p0_configs
        if p0 == 'periodic':
            self.p0 = periodic.Periodic()
        if p0 == 'exponential':
            self.p0 = exponential.Exponential()
        if p0 == 'uniform':
            self.p0 = uniform.Uniform()
        if p0 == 'normal':
            self.p0 = normal.Normal()
        if p0 == 'custom':
            self.p0 = custom.Custom()
        if p0 == 'binomial':
            self.p0 = binomial.Binomial()
        self.player_move_costs = [p0_move_cost, p1_move_cost]
        self.reset()
    
    def reset(self):
        self.p0.config(self.p0_configs)
        self.player_moves = [[0], [0]]
        self.player_total_gain = [0,0]
        self.player_total_move_cost = [0,0]
        self.p0_next_move = self.p0.first_move()
        self.controller = 0
        self.tick = 0
        self.reset_state()
        self.found_FM = False
        return self.state

    def step(self, action):
        self.tick += 1
        # if both players play, defender gets control
        if action == 1 and self.tick == self.p0_next_move:
            action = 0
        #p0 plays according to its strategy
        if self.tick == self.p0_next_move:
            self.move(0)
            self.p0_next_move = self.p0.move(self.tick)
        #p1 plays if action is 1
        if action == 1:
            self.move(1)
            if self.get_LM(0) > 0:
                self.found_FM = True
        #update output values
        self.set_state()
        reward = self.calc_rew()
        done = self.tick >= self.duration
        self.player_total_gain[self.controller] += 1
        return self.state, reward, done, {'true_action':action}

    def render(self):
        return

    def calc_rew(self):
        if self.rew_type == 'exponential':
            return calc_rew.exponential(self.found_FM,self.moved(1),self.player_move_costs[1],self.get_LM(0),self.player_moves[1],self.tick,self.rew_configs)
        elif self.rew_type == 'LM_benefit':
            return calc_rew.LM_benefit(self.found_FM,self.moved(1),self.player_move_costs[1],self.get_LM(0),self.get_LM(1),self.tick,self.player_moves[0])
        elif self.rew_type == 'New_benefit':
            return calc_rew.New_benefit(self.found_FM,self.moved(1),self.player_move_costs[1],self.controller)
        elif self.rew_type == 'reciprocal':
            return calc_rew.reciprocal(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.tick)
        elif self.rew_type == 'constant_reciprocal':
            return calc_rew.constant_reciprocal(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['upper_lim'],self.tick)
        elif self.rew_type == 'modified_reciprocal':
            return calc_rew.modified_reciprocal(self.found_FM,self.moved(1),self.state,self.player_moves[1],self.player_move_costs[1],self.rew_configs['upper_lim'],self.tick)
        elif self.rew_type == 'constant':
            return calc_rew.constant(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        elif self.rew_type == 'constant_minus_cost':
            return calc_rew.constant_minus_cost(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        elif self.rew_type == 'constant_minus_cost_norm':
            return calc_rew.constant_minus_cost_norm(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick,self.rew_configs['c'])
        elif self.rew_type == 'exp_cost':
            return calc_rew.exp_cost(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        elif self.rew_type == 'LM_avg':
            return calc_rew.LM_avg(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        else:
            raise NotImplementedError

    def reset_state(self):
        if self.state_type == 'opp_LM':
            self.state = reset_state.opp_LM()
        elif self.state_type == 'own_LM':
            self.state = reset_state.own_LM()
        elif self.state_type == 'composite':
            self.state = reset_state.composite()
        else:
            raise NotImplementedError

    def set_state(self):
        if self.state_type == 'opp_LM':
            self.state = set_state.opp_LM(self.get_LM(0),self.moved(1),self.found_FM,self.tick,self.state)
        elif self.state_type == 'own_LM':
            self.state = set_state.own_LM(self.moved(1),self.state)
        elif self.state_type == 'composite':
            self.state = set_state.composite(self.get_LM(0),self.moved(1),self.found_FM,self.tick,self.state)
        else:
            raise NotImplementedError

    def get_LM(self,player):
        return self.player_moves[player][-1]
    
    def moved(self,player):
        return self.tick == self.player_moves[player][-1]

    def move(self, player):
        self.player_moves[player].append(self.tick)
        self.player_total_move_cost[player] += self.player_move_costs[player]
        self.controller = player

    '''
    Calculate benefit
    '''
    def calc_benefit(self, player):
        return self.player_total_gain[player] - self.player_total_move_cost[player]

    def calc_avg_benefit(self, player):
        return self.calc_benefit(player, self.tick) / self.tick