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

class trainer():
    
    
    def __init__(self, num_stations, action_space, episode):
        
        self.env = env(num_stations, 50, debug = False)
        self.init_stocks = []
        self.current_stocks = self.init_stocks
        self.num_stations = num_stations
        self.action_space = action_space
        self.timestamp = self.get_timestamp(True)
        self.q_tables = []
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
        self.success_ratios = []
        self.team_cumulative_rewards = []
        
    def run(self, mode):
        
        print("==========================")
        print("start {}ing sessions ...".format(mode))
        
        self.mode = mode
        
        if self.mode == "test":
            
            # Create new env and agents; run test workflow
            self.env.eps_reset()
            self.agent_manager.eps_reset()
        
        
        for num_eps in self.eps:
            
            for eps in tqdm(range(num_eps)):
                                        
                for hour in range(0, 24):
                    
                    actions = self.agent_manager.batch_choose_action(self.current_stocks, 
                                                    self.mode, self.q_tables)
                    
                    current_hour, old_stocks, new_stocks, rewards, day_end = self.env.ping(actions)
                    
                    self.agent_manager.batch_learn(old_stocks, actions, rewards, 
                                                   new_stocks, day_end)
                    
                    self.current_stocks = new_stocks
                
                self.success_ratios.append(self.env.cal_success_ratio())
                self.team_cumulative_rewards.append(self.agent_manager.get_team_rewards())

                self.env.eps_reset()
                self.agent_manager.eps_reset()
            
            print("-------------------------")
            self.agent_manager.save_q_tables(self.timestamp)
            self.graph_performance(num_eps)
            
            if self.mode == "learn":
            
                self.q_tables = self.agent_manager.get_q_tables()

    
    def test(self):
        
        # create new stocks and agents
        
        self.env.eps_reset()
        self.agent_manager.eps_reset()
        
        # run episodes with and without transfer learning
        
        for num_eps in self.eps:
            
            for eps in tqdm(range(num_eps)):
                                        
                for hour in range(0, 24):
                    
                    actions = self.agent_manager.batch_choose_action(self.current_stocks)
                    
                    current_hour, old_stocks, new_stocks, rewards, day_end = self.env.ping(actions)
                    
                    self.agent_manager.batch_learn(old_stocks, actions, rewards, new_stocks, day_end)
                    
                    self.current_stocks = new_stocks
                
                self.success_ratios.append(self.env.cal_success_ratio())
                self.team_cumulative_rewards.append(self.agent_manager.get_team_rewards())

                self.env.eps_reset()
                self.agent_manager.eps_reset()        
        
        
    
    def graph_performance(self, num_eps):
        
        window = 100
        
        # Success Ratio
        plt.figure(figsize=(5, 4))
        title = "Success Ratio"
        x_axis = [x for x in range(num_eps)]
        plt.plot(x_axis, self.success_ratios)
        plt.xlabel("Episode")
        plt.ylabel("Group Success Ratio")
        plt.title(title)
        
        # Rolling Average of Cumulative Rewards
        plt.figure(figsize = (5, 4))
        rolling_average = self.rolling_avg(self.success_ratios, window)
        x_axis = [x for x in range(len(rolling_average))]
        plt.plot(x_axis, rolling_average)
        plt.xlabel("Episode (window = " + str(window) + ")")
        plt.ylabel("Rolling Avg. Group Success Ratio")
        plt.title("Rolling Avg. Group Success Ratio")        
        
        # Cumulative Rewards
        plt.figure(figsize=(5, 4))
        title = "Cumulative Rewards (Total of All Agents)"
        x_axis = [x for x in range(num_eps)]
        plt.plot(x_axis, self.team_cumulative_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Group Success Ratio")
        plt.title(title)
        
        # Rolling Average of Cumulative Rewards
        plt.figure(figsize = (5, 4))
        rolling_average = self.rolling_avg(self.team_cumulative_rewards, window)
        x_axis = [x for x in range(len(rolling_average))]
        plt.plot(x_axis, rolling_average)
        plt.xlabel("Episode (window = " + str(window) + ")")
        plt.ylabel("Rolling Avg. Cumulative Rewards")
        plt.title("Rolling Avg. Cumulative Rewards")
    
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
        