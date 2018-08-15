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
import pandas as pd
import numpy as np
import os

class trainer():
    
    
    def __init__(self, num_stations, action_space, episode, threshold, collaboration):
        
        # Performance Metrics
        self.success_ratios = {}
        self.team_cumulative_rewards = {}
        
        self.timestamp = self.get_timestamp(True)
        self.result_dir = "./performance_log/" + self.timestamp + "/graphs/"
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)          
        
        # Env, Agent, and Key Attributes
        self.env = env(num_stations, 50, threshold)
        self.init_stocks = []
        self.current_stocks = self.init_stocks
        self.num_stations = num_stations
        self.action_space = action_space
        self.q_tables = []
        self.merged_table = 0
        self.mode = "learn"
        self.collaboration = collaboration
        self.threshold = threshold
                
        for idx in range(num_stations):
            self.init_stocks.append(50)
        
        self.agent_manager = agent_manager(self.num_stations, 
                                           self.action_space, self.init_stocks, self.collaboration)
                
        if type(episode) == int:
            self.eps = [episode]
        
    def run(self, mode):
        
        '''
        This function runs the agent-station training and testing procedures 
        using multithreading.
        
        '''
        
        print("==========================")
        print("start {}ing sessions of {} stations ...".format(mode, self.num_stations))
        
        self.mode = mode
        self.success_ratios[mode] = []
        self.team_cumulative_rewards[mode] = []
        
        if self.mode == "test":
            
            # Create new env and agents; run test workflow
            self.env.eps_reset()
            self.agent_manager.batch_reset(self.merged_table)
        
        for num_eps in self.eps:
                        
            for eps in tqdm(range(num_eps)):
                
                upload_flag = False
            
                if num_eps % 500 == 0 and self.collaboration: 
                    upload_flag = True
                                        
                for hour in range(0, 24):
                    
                    # Agent-Station Interaction with multi-threading
                    
                    actions = self.agent_manager.batch_choose_action(self.current_stocks)
                    
                    current_hour, old_stocks, new_stocks, rewards, day_end = self.env.ping(actions)
                    
                    self.agent_manager.batch_learn(old_stocks, actions, rewards, 
                                                   new_stocks, day_end, upload = upload_flag)
                    
                    self.current_stocks = new_stocks
                
                self.success_ratios[self.mode].append(self.env.cal_success_ratio())
                self.team_cumulative_rewards[self.mode].append(self.agent_manager.get_team_rewards())

                self.env.eps_reset()
                self.agent_manager.eps_reset()
            
            print("-------------------------")
            self.agent_manager.save_q_tables(self.timestamp)
            
            if self.mode == "learn":
            
                self.q_tables, self.merged_table = self.agent_manager.get_q_tables()                
    
    def graph_performance(self, num_eps):
        
        window = num_eps / 200
                
        # Rolling Average of Group Success Ratio
        fig = plt.figure(figsize = (10, 8))
        
        learn_rolling_average = pd.Series(self.success_ratios["learn"]).rolling(window).mean()
        test_rolling_average = pd.Series(self.success_ratios["test"]).rolling(window).mean()
    
        x_axis = [x for x in range(len(learn_rolling_average))]
        plt.plot(x_axis, learn_rolling_average, label = "without TL")
        plt.plot(x_axis, test_rolling_average, label = "with TL")
        plt.xlabel("Episode (window = " + str(window) + ")")
        plt.ylabel("Rolling Avg. Group Success Ratio")
        plt.title("Rolling Avg. Group Success Ratio - "+str(len(self.success_ratios["learn"])) + " eps")        
        plt.legend()
        
        fig.savefig(self.result_dir + "/ra_success_ratio")
        
        # Rolling Variance of Group Success Ratio
        fig = plt.figure(figsize = (10, 8))
        
        learn_rolling_average = pd.Series(self.success_ratios["learn"]).rolling(window).var()
        test_rolling_average = pd.Series(self.success_ratios["test"]).rolling(window).var()
    
        x_axis = [x for x in range(len(learn_rolling_average))]
        plt.plot(x_axis, learn_rolling_average, label = "without TL")
        plt.plot(x_axis, test_rolling_average, label = "with TL")
        plt.xlabel("Episode (window = " + str(window) + ")")
        plt.ylabel("Rolling Group Success Variance")
        plt.title("Rolling Group Success Variance - "+str(len(self.success_ratios["learn"])) + " eps")        
        plt.legend()
        
        fig.savefig(self.result_dir + "/ra_success_var")
        
        
        # Rolling Average of Cumulative Rewards
        fig = plt.figure(figsize = (10, 8))
        
        learn_rolling_average = pd.Series(self.team_cumulative_rewards["learn"]).rolling(window).mean()
        test_rolling_average = pd.Series(self.team_cumulative_rewards["test"]).rolling(window).mean()

        x_axis = [x for x in range(len(learn_rolling_average))]
        plt.plot(x_axis, learn_rolling_average, label = "without TL")
        plt.plot(x_axis, test_rolling_average, label = "with TL")
        plt.xlabel("Episode (window = " + str(window) + ")")
        plt.ylabel("Rolling Avg. Cumulative Rewards")
        plt.title("Rolling Avg. Cumulative Rewards - "+str(len(self.success_ratios["test"])) + " eps")
        plt.legend()

        fig.savefig(self.result_dir + "/ra_rewards")

        fname = self.result_dir + "/session_results.txt"
        
        with open(fname, 'w') as f:
            
            # Transfer Learning Ratio
    
            area_wo_TL = np.array(self.team_cumulative_rewards["learn"]).sum()
            area_TL = np.array(self.team_cumulative_rewards["test"]).sum()
        
            r = (area_TL - area_wo_TL)/area_wo_TL
            
            f.write("Number of Episodes: {}".format(num_eps))
            f.write("\n")
            f.write("Threshold: {}".format(self.threshold))
            f.write("\n")
            f.write("Action Space: {}".format(self.action_space))
            f.write("\n")  
            f.write("Collaboration: {}".format(self.collaboration))
            f.write("\n")
            f.write("Reward Area without TL: {}".format(area_wo_TL))
            f.write("\n")
            f.write("Reward Area with TL: {}".format(area_TL))
            f.write("\n")
            f.write("Ratio of Trasfer Learning: {0:.2f}".format(r))
            f.write("\n")
            
            # Compare Total Team Success Count
            
            cnt_team_success_wo_TL = sum(i == 1.0 for i in self.success_ratios["learn"])
            cnt_team_success_TL = sum(i == 1.0 for i in self.success_ratios["test"])
            
            f.write("Count of Team Success without TL: {}".format(cnt_team_success_wo_TL))
            f.write("\n")
            f.write("Count of Team Success with TL: {}".format(cnt_team_success_TL))
            
            
        # Plot Bike Moving Cost of Successful Network
        cost = []
        
        for idx, val in enumerate(self.success_ratios["learn"]):
            
            if val == 1.0:
                cost.append(np.abs(self.team_cumulative_rewards["learn"][idx]) - 50.0)
        
        rolling_cost = pd.Series(cost).rolling(window).mean()
        
        fig = plt.figure(figsize = (10, 8))
        x_axis = [x for x in range(len(rolling_cost))]
        plt.plot(x_axis, rolling_cost)
        plt.xlabel('Complete Network Success Incidence ordered by Episodes')
        plt.ylabel("Rolling Cost (Total Reward - Success Reward)")
        plt.title("Cost of Bike Moving for Complete Network Success (window = " + str(window) + ")")
        
        fig.savefig(self.result_dir + "/rolling_cost")
        pd.Series(cost).to_csv(self.result_dir + "/rolling_cost.csv")
                
    
    def get_timestamp(self, replace):
        
        '''
        This function returns a timestamp or a concatenated form of it.
        
        '''
        
        if replace == True:
        
            return str(datetime.datetime.now()).replace(" ", "").replace(":", "").\
                        replace(".", "").replace("-", "")
        
        else:
            
            return str(datetime.datetime.now())
        