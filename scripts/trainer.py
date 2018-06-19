#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:36:11 2018

@author: Ian Xiao
"""

from env import env
from rl_brain import agent_manager
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import numpy as np
import pandas as pd

class trainer():
    
    
    def __init__(self, num_stations, action_space, episode):
        
        self.env = env(num_stations, 50, debug = False)
        self.init_stocks = []
        self.current_stocks = self.init_stocks
        self.num_stations = num_stations
        self.action_space = action_space
        self.timestamp = self.get_timestamp(True)
        self.q_tables = []
        self.merged_table = 0
        self.mode = "learn"
                
        for idx in range(num_stations):
            self.init_stocks.append(50)
        
        self.agent_manager = agent_manager(self.num_stations, 
                                           self.action_space, self.init_stocks)
                
        if type(episode) == int:
            self.eps = [episode]

        elif type(episode) == list:
            episode_list = [eps for eps in range(episode[0], episode[1]+1, episode[2])]
            self.eps = episode_list
        
        # Performance Metrics
        self.success_ratios = {}
        self.team_cumulative_rewards = {}
        
    def run(self, mode):
        
        print("==========================")
        print("start {}ing sessions ...".format(mode))
        
        self.mode = mode
        self.success_ratios[mode] = []
        self.team_cumulative_rewards[mode] = []
        
        if self.mode == "test":
            
            # Create new env and agents; run test workflow
            self.env.eps_reset()
            self.agent_manager.batch_reset(self.merged_table)
        
        for num_eps in self.eps:
            
            for eps in tqdm(range(num_eps)):
                                        
                for hour in range(0, 24):
                    
                    actions = self.agent_manager.batch_choose_action(self.current_stocks)
                    
                    current_hour, old_stocks, new_stocks, rewards, day_end = self.env.ping(actions)
                    
                    self.agent_manager.batch_learn(old_stocks, actions, rewards, 
                                                   new_stocks, day_end)
                    
                    self.current_stocks = new_stocks
                
                self.success_ratios[self.mode].append(self.env.cal_success_ratio())
                self.team_cumulative_rewards[self.mode].append(self.agent_manager.get_team_rewards())

                self.env.eps_reset()
                self.agent_manager.eps_reset()
            
            print("-------------------------")
            self.agent_manager.save_q_tables(self.timestamp)
            
            if self.mode == "learn":
            
                self.q_tables, self.merged_table= self.agent_manager.get_q_tables()                
    
    def graph_performance(self, num_eps):
        
        window = num_eps / 20
        print(self.success_ratios.keys())
        
        # Rolling Average of Group Success Ratio
        plt.figure(figsize = (5, 4))
        
        learn_rolling_average = self.rolling_avg(self.success_ratios["learn"], window)
        test_rolling_average = self.rolling_avg(self.success_ratios["test"], window)
        x_axis = [x for x in range(len(learn_rolling_average))]
        plt.plot(x_axis, learn_rolling_average, label = "without TL")
        plt.plot(x_axis, test_rolling_average, label = "with TL")
        plt.xlabel("Episode (window = " + str(window) + ")")
        plt.ylabel("Rolling Avg. Group Success Ratio")
        plt.title("Rolling Avg. Group Success Ratio")        
        plt.legend()
        
        
        # Rolling Average of Cumulative Rewards
        plt.figure(figsize = (5, 4))
        learn_rolling_average = self.rolling_avg(self.team_cumulative_rewards["learn"], window)
        test_rolling_average = self.rolling_avg(self.team_cumulative_rewards["test"], window)
        x_axis = [x for x in range(len(learn_rolling_average))]
        plt.plot(x_axis, learn_rolling_average, label = "without TL")
        plt.plot(x_axis, test_rolling_average, label = "with TL")
        plt.xlabel("Episode (window = " + str(window) + ")")
        plt.ylabel("Rolling Avg. Cumulative Rewards")
        plt.title("Rolling Avg. Cumulative Rewards")
        plt.legend()
    
    def rolling_avg(self, rewards, n):
        ret = np.cumsum(rewards, dtype = float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
        
    
    def get_timestamp(self, replace):
        
        if replace == True:
        
            return str(datetime.datetime.now()).replace(" ", "").replace(":", "").\
                        replace(".", "").replace("-", "")
        
        else:
            
            return str(datetime.datetime.now())
        