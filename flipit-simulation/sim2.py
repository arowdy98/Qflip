import gym
import gym_flipit
import numpy as np
import pprint
import random
from strategies import Q_Net2, Q_Net3, Greedy_Max
import matplotlib.pyplot as mlt

class Simulation:
    
    def run(self,debug,outfile,p1_strategy,p1_config,p1_cost,p0_strategy,p0_config,p0_cost,duration,rew_type,rew_config,obs_type,run_id=0):
        # set up environment and players
        env = gym.make('FlipNet-v0')
        if rew_type == 'constant_minus_cost_norm':
            rew_config['val'] = p0_config['avg_mv']
        if obs_type == "composite":
            p1_config["obs_size"] *= 2
        env.config(obs_type,rew_type,rew_config,p0_strategy,p0_config,duration,p0_cost,p1_cost)
        observation = env.reset()
        a = self.gen_attacker(p0_config['n_node'],p1_strategy,env,p1_config,p1_cost,debug)
        with open(outfile, 'w') as f:
            f.write('tick,p1_benefit,p0_benefit\n')

        # print info
        if debug:
            print('---\nRun {}'.format(run_id))
            print('attacker config ({}):'.format(p1_strategy))
            print('attacker rew type:',rew_type)
            print('attacker cost:', p1_cost)
            pprint.pprint(p1_config)
            print('defender config ({}):'.format(p0_strategy))
            pprint.pprint(p0_config)

        benefit_per_100 = []
        prev_benefit = 0

        for tick in range(duration):
            prev_observation = observation.copy()
            # prev_encoded_obs = encoded_obs.copy()
            
            action = a.pre(tick,prev_observation, duration)
            # print(action)
            observation, reward, done, info = env.step(action)
            # encoded_obs = self.update_observation(encoded_obs, observation, obs_type)
            a.post(tick,prev_observation,observation,reward,action,info['true_action'])
            if tick % 100 == 0:
                benefit_per_100.append(env.calc_benefit(1)-prev_benefit)
                prev_benefit = env.calc_benefit(1)
                with open(outfile, 'a') as f:
                    f.write('{},{},{}\n'.format(tick,env.calc_benefit(1),env.calc_benefit(0)))

            if done:
                if debug:
                    print('Attacker: {}\nDefender:{}'.format(p1_strategy,p0_strategy))
                    print('Total p1 benefit: {}'.format(env.calc_benefit(1)))
                    print('Total p0 benefit: {}'.format(env.calc_benefit(0)))
                    print("\n\n\n")
                break
        
        total_terms = int(len(benefit_per_100))
        print("Average Benefit per 100 Ticks:{}".format(sum(benefit_per_100[total_terms-100:])/(100)))
        
        avg_ben = []
        for i in range(len(benefit_per_100)):
            if i < 9:
                avg_ben.append(sum(benefit_per_100[:i+1])/(i+1))
                continue
            avg_ben.append(sum(benefit_per_100[i-9:i+1])/10)

        with open('summary.csv', 'w') as f:
            f.write('episode,p1_benefit_{}_{}_{}\n'.format(p1_strategy,rew_type,p0_strategy))
            for i in range(len(avg_ben)):
                f.write('{},{}\n'.format(i,avg_ben[i]))
            f.write('Average benefit,{}\n'.format(sum(benefit_per_100[total_terms-100:])/(100)))

        # mlt.plot(benefit_per_100, label = rew_type + " " + p1_strategy)
        mlt.plot(avg_ben, label = rew_type + " " + p1_strategy + "average")
        mlt.ylabel("Benefit Per Episode",fontsize=14)
        mlt.xlabel("Episodes",fontsize=14)
        mlt.legend()
        mlt.show()

    def gen_attacker(self,n_node,s,env,p1_config,p1_cost,debug):
        if 'q-net2' in s:
            return Q_Net2.Q_Net2(n_node,2,p1_config,debug=debug)
        elif 'q-net3' in s:
            return Q_Net3.Q_Net3(n_node,2,p1_config,debug=debug)
        elif 'greedy-max' in s:
            return Greedy_Max.Greedy_Max(n_node,p1_config)
        else:
            raise NotImplementedError

    def update_observation(self, encoded_obs, observation, obs_type):
        for i in range(len(encoded_obs)):
            if encoded_obs[i] != -1:
                encoded_obs[i] += 1

        if obs_type == "composite":
            if (observation[1] != -1) and (observation[1] != (encoded_obs[-1])):
                del encoded_obs[0]
                del encoded_obs[0]
                encoded_obs.append(observation[0]/200)
                encoded_obs.append(observation[1]/200)

        else:
            if (observation != -1) and (observation != (encoded_obs[-1])):
                del encoded_obs[0]
                encoded_obs.append(observation/200)
        return encoded_obs
