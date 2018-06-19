#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:50:38 2018

@author: Ian Xiao
"""

import numpy as np
import pandas as pd
import os

class agent_manager():
    
    '''
    this class handles the interaction with environment, manages RL agents, 
    and maintains the Transfer Learning process
    
    '''
    
    def __init__(self, num_agent, action_space, init_stock_list):
        
        self.num_agent = num_agent
        self.action_space = action_space
        self.init_stock = init_stock_list
        self.agent_list = self.init_agents()
        self.mode = "learn"
        
    def init_agents(self, trained_table = None):
        
        print("Generating agents ...")
        
        if trained_table is not None:
                    
            agent_list = []
            
            for idx in range(self.num_agent):
                
                agent_list.append(agent(idx, self.action_space, epsilon = 0.9, lr = 0.01, 
                                      gamma = 0.9, 
                                      current_stock = self.init_stock[idx],
                                      trained_table = trained_table))
                
        else: 
            
            agent_list = []
            
            for idx in range(self.num_agent):
                
                agent_list.append(agent(idx, self.action_space, epsilon = 0.9, lr = 0.01, 
                                      gamma = 0.9, 
                                      current_stock = self.init_stock[idx]))
            
        
        return agent_list
        
    def ping_env(self):
        
        actions = []
        
        print("pinging environment")
    
        return actions
        
    def batch_learn(self, s, a, r, s_, day_end):
        
        '''
        This function updates Q tables and Meta Q Table after each interaction 
        with the environment.
        Input: 
            - s: current bike stock
            - a: current action (number of bikes to move)
            - r: reward received from current state
            - s_: new bike stock based on bike moved and new stock
            - g: game over flag
        
        '''
                
        for idx in range(self.num_agent):
            self.agent_list[idx].learn(s[idx], a[idx], r[idx], s_[idx], day_end)
                        
            
    def batch_choose_action(self, s):
        
        actions = []
        
        for idx in range(self.num_agent):
            action = self.agent_list[idx].choose_action(s[idx])
                        
            actions.append(action)
            
        return actions
        
    def batch_reset(self, q_talbe):
        
        self.agent_list = self.init_agents(q_talbe)
        
    def get_team_rewards(self):
        
        team_rewards = []
        
        for agent in self.agent_list:
            team_rewards.append(agent.get_rewards())
            
        return sum(team_rewards)
    
    
    def get_q_tables(self):
        
        # Collect Q Tables from All Agents
        q_tables = []

        for agent in self.agent_list:
            q_tables.append(agent.get_q_table())
        
            
        # Concatenate all Q Tables
        merged_table = q_tables[0]
        
        if len(q_tables) > 1:
            
            for idx in range(1, len(q_tables)): 
                
                merged_table = pd.concat([merged_table, q_tables[idx]], axis = 0)           
        
        # De-duplicate Merged Q Tables by Grouping and Averaging the Values
        
        merged_table = merged_table.groupby(merged_table.index)[merged_table.columns.values].mean()
        
        merged_table.fillna(0.0, inplace = True)
        
        return q_tables, merged_table
        
            
    def eps_reset(self):
            
        for agent in self.agent_list:
            agent.reset_cumulative_reward()
            
    def save_q_tables(self, timestamp):
                
        dir_path = "./performance_log/" + timestamp + "/q_tables/"
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)        
        
        for agent in self.agent_list:
            q_table = agent.get_q_table()
            q_table.to_csv(dir_path + agent.get_name() + "_q_table.csv")
        

'''
-------------------------------------------------------------------------------
'''
        
class agent():
    
    '''
    this is a class object for the RL agent, which include learning
    and decisioning
    
    '''
    
    def __init__(self, name, action_space, epsilon, 
                 lr, gamma, current_stock, trained_table = None):
        
        self.name = "a" + str(name)
        self.actions = action_space
        self.reward = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.current_stock = current_stock
        
        if trained_table is not None: 
            
            self.q_table = trained_table
            print("{}: hello :)  I am ready with some knowledge.".format(self.name))
            
        else: 
            
            self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)
            print("{}: hello :)  I am ready, but new to this.".format(self.name))
        
        
        # performance metric
        self.hourly_action_history = []
        self.hourly_stock_history = []
        self.cumulative_reward = 0.0        
        self.check_state_exist(current_stock)

        
    def choose_action(self, s):
        
        '''
        This funciton choose an action based on Q Table. It also does 
        validation to ensure stock will not be negative after moving bikes.
        Input: 
            - s: current bike stock
            - ex: expected bike stock in subsequent hour (based on random forests prediction)
        
        Output:
            - action: number of bikes to move
        
        '''
        
        self.check_state_exist(s)
        self.current_stock = s
        
        # find valid action based on current stock 

        valid_state_action = self.q_table.loc[s, :]
                
        if np.random.uniform() < self.epsilon:
        
            try:
                state_actions = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                action = state_actions.idxmax()
                
            except:
                action = 0
                                              
        else:
            
            # randomly choose an action
            # re-pick if the action leads to negative stock
            try:
                action = np.random.choice(valid_state_action.index)
                
            except:
                action = 0
        
        self.hourly_action_history.append(action)
        self.hourly_stock_history.append(s)
        
        return action
        
        
    def learn(self, s, a, r, s_, day_end):

        
        '''
        This function updates Q tables after each interaction with the
        environment.
        Input: 
            - s: current bike stock
            - ex: expected bike stock in next hour
            - a: current action (number of bikes to move)
            - r: reward received from current state
            - s_: new bike stock based on bike moved and new stock
        Output: None
        '''
        
        self.check_state_exist(s_)
        
        q_predict = self.q_table.loc[s, a]
        
        if day_end == False:
            # Updated Q Target Value if it is not end of day  
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        
        else:
            # Update Q Target Value as Immediate reward if end of day
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        
        self.cumulative_reward = r + self.cumulative_reward
        
        return

        
    def check_state_exist(self, state):
        
        # Add a new row with state value as index if not exist
        if state not in self.q_table.index:
            
            self.q_table = self.q_table.append(
                pd.Series(
                        [0]*len(self.actions), 
                        index = self.q_table.columns,
                        name = state
                        )
                )
        
    def print_q_table(self):
        
        print(self.q_table)


    def get_q_table(self):
        
        return self.q_table

    
    def get_hourly_actions(self):
        
        return self.hourly_action_history
    
        
    def get_hourly_stocks(self):
        
        return self.hourly_stock_history
        
        
    def get_rewards(self):
        
        return self.cumulative_reward
        
        
    def reset_cumulative_reward(self):
        
        self.cumulative_reward = 0.0
        
    def get_name(self):
        
        return str(self.name)
        
    
        
'''
-------------------------------------------------------------------------------
'''
        
class TL():
    
    '''
    this is a class for the Transfer Learning mechnism.
    
    '''
    
    def __init__(self):
        
        self.name = "this is a TL object"