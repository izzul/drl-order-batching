'''
Code for defining the simulation environment into a gym environment
'''
import gym
import pandas as pd
import numpy as np
import yaml

from sim_model.whOptim import whOptim
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class WAREHOUSE(gym.Env):

    def __init__(self, params=(1, 1)):

        # Open file with parameters
        with open(r'config/scenario_tikno_1.yml') as file:
            config = yaml.full_load(file)

        self.config = config
        self.params = params

        # Set Gym variables for action space and observation space
        self.action_space = gym.spaces.Discrete(self.config['environment']['action_space'])
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.config['environment']['observation_space'],), dtype=np.uint8)

        # Set simulation parameters and load order data
        self.batching_opt = self.config['simulation']['batching_opt']
        self.routing_opt = self.config['simulation']['routing_opt']
        self.picker_number = self.config['simulation']['picker_number']
        self.cart_capacity = self.config['simulation']['cart_capacity']

        # Initiate simulation instance
        self.sim = whOptim(self.batching_opt, self.routing_opt, self.picker_number, self.cart_capacity)

        # Set parameters to capture simulation performance
        self.steps = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_reward_copy = 0
        self.episode_reward_hist = -20000
        self.infeasible_actions = []
        self.infeasible_ratio = 0
        self.processed_order = 0
        self.tardy_order = 0
        self.cart_utility = 0
        self.picker_utility = 0
        self.lateness = 0

        self.avg_completion_time = 0
        self.avg_turn_over_time = 0
    # Reset function for agent, saves information and reinitiates simulation instance
    def reset(self):
        # Print and save information about the processed episode
        print('Episode {0} finished, steps per episode: {1}'.format(self.episode, self.episode_step))
        self.sim.episode_render(self.episode)

        self.avg_completion_time = self.sim.last_state[8]
        self.avg_turn_over_time = self.sim.last_state[9]
        # Initiate simulation environment and get initial state
        self.sim = whOptim(self.batching_opt, self.routing_opt, self.picker_number, self.cart_capacity)
        state = self.sim.get_state()
        self.episode += 1
        self.episode_step = 0
        self.episode_reward_hist = self.episode_reward_copy
        self.episode_reward_copy = 0
        return np.array(state)

    def step(self, action):
        #print('eps step: %d', self.episode_step)
        # Check whether this is a feasible action
        state = self.sim.get_state()
        feasibility = self.sim.check_action(action, state)

        # Compute reward based on current state and action pair
        reward = self.sim.get_reward(action)

        # If the reward is feasible, simulate the action in the simulation model and observe new state
        # If the reward is not feasible, do nothing and provide negative reward
        if feasibility:
            state = self.sim.simulate(action)
            self.infeasible_actions.append(1)
        else:
            self.infeasible_actions.append(0)

        # Check whether an episode is done. An episode is done when all orders have been processed
        done = self.sim.check_termination()

        # If at some point, the agent takes too many step, the episode is terminated
        if self.episode_step > 400000:
            done = True
            print('Episode stopped, {0} steps taken'.format(self.episode_step))

        # If an episode is done, save information
        if done:
            self.sim.reward_distribution['final_reward'] += (1 - (self.sim.trigger.tardy_order + len(self.sim.trigger.current_pool[0][2])) / self.sim.trigger.total_order)**2
            reward += ((1 - (self.sim.trigger.tardy_order + len(self.sim.trigger.current_pool[0][2])) / self.sim.trigger.total_order) ** 2) * self.params[0]
            print('final episode reward: %s' % (str(reward)))

        # Save information based on the step that was taken
        self.episode_reward_copy += reward
        self.steps += 1
        self.episode_step += 1
        self.infeasible_ratio = self.infeasible_actions.count(0) / len(self.infeasible_actions)
        self.processed_order = state[0]
        self.tardy_order = state[5]
        self.cart_utility = state[10]
        self.picker_utility = state[4]
        self.lateness = [11]
        if len(self.infeasible_actions) > 1000:
            self.infeasible_actions = []

        return np.array(state), reward, done, {}

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError()
        return np.array([0, 0, 0])

