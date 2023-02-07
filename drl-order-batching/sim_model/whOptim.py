import time
import simpy
import pandas as pd
import random
from datetime import timedelta
from .modules import general_function as gf

from .modules import triggers
from .modules import batchings
from .modules import routings
# Reading order file
class whOptim:
    def __init__(self, batching_opt, routing_opt, picker_number, cart_capacity):
        # prepare simulation opt
        self.batching_opt = batching_opt
        self.routing_opt = routing_opt
        self.picker_number = picker_number
        self.cart_capacity = cart_capacity
        self.order_file = gf.reading_file()[0]

        # Specify urgent threshold in seconds
        self.urgent_threshold = 300
        
        # prepare simulator
        self.env = simpy.Environment()
        self.picker_pool = simpy.Resource(self.env, capacity=picker_number)
        self.routing = routings.routings(routing_opt)
        self.batching = batchings.batchings(batching_opt, cart_capacity, self.routing, self.urgent_threshold)
        self.trigger = triggers.triggers(self.env, self.picker_pool, self.batching, self.routing, self.cart_capacity, self.urgent_threshold)
        self.trigger.prepare(self.order_file)
        self.env.process(self.trigger.step())
        # Counting total order
        self.total_order = list()
        # Counting total item picked
        self.total_item_picked = list()

        self.order_list = list()
        self.trigger_list = list()
        self.batching_list = list()
        self.routing_list = list()
        self.picker_list = list()
        self.cart_list = list()

        self.total_completion_time = list()
        self.average_completion_time = list()
        self.total_turn_over_time = list()
        self.average_turn_over_time = list()
        self.average_picker_utility = list()
        self.average_cart_utility = list()
        self.total_tardy_order = list()
        self.total_lateness = list()
        self.average_lateness = list()
        self.total_batches = list()

        self.time_limit = timedelta(hours=8).seconds

        self.reward_episode = 0
        self.last_state = [0] * 14

        self.reward_structure = {'infeasible_action': -0.5, 'tardy_order': -1.5, 'feasible_action': 0,
                                 'batch_action': 0.1}
        self.reward_distribution = {'infeasible_action': 0, 'tardy_order': 0, 'batch_action': 0, 'batch_composition': 0,
                                    'final_reward': 0}
        self.weight_tardy = 1

    def get_state(self):
        average_completion_time = 0
        average_turn_over_time = 0
        average_cart_utility = 0
        average_lateness = 0
        if (self.trigger.processed_order > 0):
            # Listing average completion time
            average_completion_time = (round(self.trigger.completion_time.total_seconds()/60/(self.trigger.processed_order), 2))
            # Listing avg Turonver time
            average_turn_over_time = (round(self.trigger.turn_over_time.total_seconds()/60/(self.trigger.processed_order), 2))
            # Counting & listing average cart utility
            ave_cart_utility = round(self.trigger.cart_utility/self.trigger.total_batch, 2)
            average_cart_utility = (ave_cart_utility)
            # Listing Average Lateness
            average_lateness = (round(self.trigger.total_lateness.total_seconds()/(self.trigger.processed_order)/60,2))

        order_pool_content = len(self.trigger.current_pool[0][2])
        available_picker = self.picker_pool.capacity - self.picker_pool.count
        state_rep = [
            self.trigger.processed_order, # 0
            self.trigger.processed_item, # 1
            round(self.trigger.completion_time.total_seconds()/60, 2), # 2
            round(self.trigger.turn_over_time.total_seconds()/60, 2), # 3
            round(self.trigger.completion_time/(timedelta(hours=8)*self.picker_number), 2), # 4
            self.trigger.tardy_order + len(self.trigger.current_pool[0][2]), # 5
            round(self.trigger.total_lateness.total_seconds()/60, 2), # 6
            self.trigger.total_batch, # 7
            average_completion_time, # 8
            average_turn_over_time, # 9
            average_cart_utility, # 10
            average_lateness, # 11
            order_pool_content, # 12
            available_picker # 13
        ]
        self.last_state = state_rep
        return state_rep

    def simulate(self, action):
        # run simulation second by second
        self.trigger.action = action
        self.env.run(until=self.env.now+1)
        return self.get_state()
        # if (self.trigger.print_io_station):
        #     dfIoS = pd.DataFrame(self.trigger.print_ioStation, columns =['TriggerID', 'OrderIDs', 'BatchID', 'OrderCount', 'StartingTime', 'FinishingTime'])
        #     pd.set_option('display.max_rows', None)
        #     print('IoStation Content')
        #     print(dfIoS)

    def episode_render(self, episode_num):
        self.order_list.append(1)
        self.trigger_list.append('DRL')
        self.batching_list.append(self.batching_opt)
        self.routing_list.append(self.routing_opt)
        self.picker_list.append(self.picker_number)
        self.cart_list.append(self.cart_capacity)

        # Listing total order
        self.total_order.append(self.trigger.processed_order)
        # Listing total total time
        self.total_item_picked.append(self.trigger.processed_item)
        # Listing total completion time
        self.total_completion_time.append(round(self.trigger.completion_time.total_seconds()/60, 2))
        # Listing total Turonver time
        self.total_turn_over_time.append(round(self.trigger.turn_over_time.total_seconds()/60, 2))
        # Counting and listing average picker utility
        self.average_picker_utility.append(round(self.trigger.completion_time/(timedelta(hours=8)*self.picker_number), 2))
        # Listing total on time delivery
        self.total_tardy_order.append(self.trigger.tardy_order + len(self.trigger.current_pool[0][2]))
        # Listing Total Lateness
        self.total_lateness.append(round(self.trigger.total_lateness.total_seconds()/60, 2))
        # Listing Total Batches Processed
        self.total_batches.append(self.trigger.total_batch)

        if (self.trigger.processed_order > 0):
            # Listing average completion time
            self.average_completion_time.append(round(self.trigger.completion_time.total_seconds()/60/(self.trigger.processed_order), 2))
            # Listing avg Turonver time
            self.average_turn_over_time.append(round(self.trigger.turn_over_time.total_seconds()/60/(self.trigger.processed_order), 2))
            # Counting & listing average cart utility
            ave_cart_utility = round(self.trigger.cart_utility/self.trigger.total_batch, 2)
            self.average_cart_utility.append(ave_cart_utility)
            # Listing Average Lateness
            self.average_lateness.append(round(self.trigger.total_lateness.total_seconds()/(self.trigger.processed_order)/60,2))
        else:
            self.average_completion_time.append(0)
            self.average_turn_over_time.append(0)
            self.average_cart_utility.append(0)
            self.average_lateness.append(0)

        # Put result on file
        result = pd.DataFrame({
            'OrderFile': self.order_list,
            'TriggerMethod': self.trigger_list,
            'BatchingMethod': self.batching_list,
            'RoutingPolicy': self.routing_list,
            'NumOfPickers': self.picker_list,
            'CartCapacity': self.cart_list,
            'TotalOrder': self.total_order,
            'TotalBatch': self.total_batches,
            'CompletionTime': self.total_completion_time,
            'AvgCompletionTime': self.average_completion_time,
            'TurnOverTime': self.total_turn_over_time,
            'AvgTurnOverTime': self.average_turn_over_time,
            'TotalItemPicked': self.total_item_picked,
            'AvgPickerUtil': self.average_picker_utility,
            'AvgCartUtil': self.average_cart_utility,
            'NumOfLate': self.total_tardy_order,
            'TotalLateness': self.total_lateness,
            'AvgLateness': self.average_lateness
            }, columns = ['OrderFile','TriggerMethod','BatchingMethod','RoutingPolicy','NumOfPickers','CartCapacity','TotalOrder','TotalBatch','CompletionTime','AvgCompletionTime','TurnOverTime', 'AvgTurnOverTime','TotalItemPicked','AvgPickerUtil','AvgCartUtil','NumOfLate','TotalLateness','AvgLateness'])
        if episode_num == 0:
            result.to_csv('All.csv')
        else:
            result.to_csv('All.csv', mode='a', header=False)
        print(result)
        self.reward_episode = 0

    def check_action(self, action, state=None):
        if action == 0:
            # cannot trigger batching if current_pool is empty
            if self.trigger.current_pool[0][1] == 0:
                return False
            # cannot trigger batching if no available picker
            elif self.picker_pool.count == self.picker_pool.capacity:
                return False
            return True
        else:
            # cannot be idle if there is urgent order
            if self.trigger.check_urgent():
                return False
            return True

    def get_reward(self, action):
        self.reward_action = 0
        feasibility = self.check_action(action)
        tardy_orders = self.last_state[11]

        if not feasibility:
            self.reward_action += self.reward_structure['infeasible_action']
            self.reward_distribution['infeasible_action'] += self.reward_structure['infeasible_action']

        if tardy_orders > 0:
            self.reward_action += (self.reward_structure['tardy_order'] * tardy_orders) * self.weight_tardy
            self.reward_distribution['tardy_order'] += self.reward_structure['tardy_order'] * tardy_orders

        if action == 0:
            # self.reward_action += self.reward_structure['batch_action'] * self.weight_picking
            self.reward_distribution['batch_action'] += self.reward_structure['batch_action']
            # self.reward_distribution['batch_composition'] -= (1 - self.nOrders_hist / self.max_batchsize_ptg_gtp)/5

        self.reward_episode += self.reward_action
        return self.reward_action

    def check_termination(self):
        if self.trigger.current_row < self.trigger.total_order and self.env.now <= self.time_limit:
            return False
        return True
