import gym
import gym_flipit
import numpy as np
import pprint
from strategies import GreedyWrapper,Q,Renewal,PeriodicOptimal,Q_Batch, Q_Net
import matplotlib.pyplot as mlt

class Simulation:
    # debug (bool): print info during play
    # p1_strategy (str): agent strategy (q-table, greedy, optimal-periodic, renewal)
    # p1_config (dict): config values for agent (values depend on strategy)
    # p1_cost (non-negative int): move cost for p1
    # p0_strategy (str): renewal opponent strategy (periodic, exponential, normal, uniform)
    # p0_config (dict): parameters for renewal opponent's distribution
    # p0_cost (non-negative int): move cost for p0
    # duration (int): game duration in ticks
    # rew_type (string): name of reward type used in gym env (constant_minus_cost_norm, constant_minus_cost, LM_benefit, exponential, reciprocal, constant_reciprocal, constant, exp_cost, LM_avg)
    # rew_config (dict): parameters for chosen reward type
    # obs_type (string): name of observation scheme used in gym env (opp_LM,own_LM,composite)
    # run_id (any): identifier for given run (OPTIONAL)

    def run(self,debug,outfile,p1_strategy,p1_config,p1_cost,p0_strategy,p0_config,p0_cost,duration,rew_type,rew_config,obs_type,run_id=0):
        # set up environment and players
        env = gym.make('Flipit-v0')
        if rew_type == 'constant_minus_cost_norm':
            rew_config['val'] = p0_config['avg_mv']
        env.config(obs_type,rew_type,rew_config,p0_strategy,p0_config,duration,p0_cost,p1_cost)
        observation = env.reset()
        a = self.gen_attacker(p1_strategy,env,p1_config,p1_cost,debug)
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
        # acts = []
        # prev_tick = 0
        # run simulation
        benefit_per_100 = []
        prev_benefit = 0
        for tick in range(duration):
            prev_observation = observation
            action = a.pre(tick,prev_observation)
            # if action == 1:
            #     acts.append(tick-prev_tick)
            #     prev_tick = tick
                # print("Defender:",tick)
            observation, reward, done, info = env.step(action)
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
                    # print(env.player_moves[0])
                    # print(env.player_moves[1])
                    # print(a.T)
                    print("\n\n\n")
                break
        if p0_config == "custom":
            j = 1.0
            mv_freq = 0
            for i in p0_config['dist']:
                mv_freq += i*j
                j += 1
            print("True average moving frequency of opponent:",mv_freq)

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
        mlt.ylabel("Benefit Per Episode")
        mlt.xlabel("Episodes")
        mlt.legend()
        mlt.show()
        # mlt.plot(p0_config['dist'])
        # mlt.ylabel("Probability of moving")
        # mlt.xlabel("Time step between consecutive moves")
        # mlt.show()

    def gen_attacker(self,s,env,p1_config,p1_cost,debug):
        if 'greedy' in s:
            return GreedyWrapper.GreedyWrapper(env.p0,p1_cost,debug=debug)
        elif 'q-table' in s:
            return Q.Q(env.action_space.n,p1_config,debug=debug)
        elif 'q-batch' in s:
            return Q_Batch.Q_Batch(env.action_space.n,p1_config,debug=debug)
        elif 'q-net' in s:
            return Q_Net.Q_Net(env.action_space.n,p1_config,debug=debug)
        elif 'optimal-periodic' in s:
            return PeriodicOptimal.PeriodicOptimal(env.p0, p1_cost)
        elif 'renewal' in s:
            return Renewal.Renewal(s)
        else:
            raise NotImplementedError
