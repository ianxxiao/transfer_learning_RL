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
    
    def __init__(self, num_agent, action_space, init_stock_list, collaboration):
        
        self.num_agent = num_agent
        self.action_space = action_space
        self.init_stock = init_stock_list
        self.collaboration = collaboration
        
        self.agent_list = self.init_agents()
        self.knowledge_repo = pd.DataFrame()
        self.mode = "learn"

        
    def init_agents(self, trained_table = None):
        
        '''
        This function runs the agent initiation procedure.
        Output: 
            agent_list: a list of agent class object
            
        '''
        
        print("Generating agents ...")
        
        if trained_table is not None:
                    
            agent_list = []
            
            for idx in range(self.num_agent):
                
                agent_list.append(agent(idx, self.action_space, epsilon = 0.9, lr = 0.01, 
                                      gamma = 0.9, 
                                      current_stock = self.init_stock[idx],
                                      collaboration = self.collaboration,
                                      trained_table = trained_table, 
                                      ))

        else: 
            
            agent_list = []
            
            for idx in range(self.num_agent):
                
                agent_list.append(agent(idx, self.action_space, epsilon = 0.9, lr = 0.01, 
                                      gamma = 0.9, 
                                      current_stock = self.init_stock[idx], 
                                      collaboration = self.collaboration))
            
        return agent_list
        
        
    def batch_learn(self, s, a, r, s_, day_end, upload):
        
        '''
        This function updates Q tables and Meta Q Table after each interaction 
        with the environment. Multiple Thread Enabled
        Input: 
            - s: current bike stock
            - a: current action (number of bikes to move)
            - r: reward received from current state
            - s_: new bike stock based on bike moved and new stock
            - day_end: flag if end of day
            - upload: flag for uploading to Knowledge Repo
        
        '''
        
        for idx in range(self.num_agent):
            
            self.agent_list[idx].learn(s[idx], a[idx], r[idx], s_[idx], day_end)
            
        if upload == True: 
            
            self.knowledge_repo = self.get_q_tables()[1]    
                            
    def batch_choose_action(self, s):
        
        '''
        This function finds and returns the best actions based on the states.
        Input:
            - s: a list of state of all stations
        
        Output: 
            - actions: a list of best action for the corresponding state
        
        '''
        
        actions = []
        
        for idx in range(len(self.agent_list)):
        
            action = self.agent_list[idx].choose_action(s[idx], self.knowledge_repo)
            actions.append(action)      

        return actions
        
    def batch_reset(self, q_talbe):
        
        '''
        This function re-initiate agents with trained q-tables to enable 
        Transfer Learning.
        
        '''
        
        self.agent_list = self.init_agents(q_talbe)
        
    def get_team_rewards(self):
        
        '''
        This function collects rewards from each agent and returns the sum.
        
        '''
        
        team_rewards = []
        
        for agent in self.agent_list:
            team_rewards.append(agent.get_rewards())
            
        return sum(team_rewards)
    
    
    def get_q_tables(self):
        
        '''
        This function collects Q Table from each agent and aggregate to one single Q Table
        
        Outputs:
            - q_tables: a list of Q Table pandas data frame
            - merged_table: a pandas data frame of combined Q Table from all agents
        
        '''
        
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
        
        '''
        This function reset agents by clearing the cumulative rewards.
        
        '''
            
        for agent in self.agent_list:
            agent.reset_cumulative_reward()
            
    def save_q_tables(self, timestamp):
        
        '''
        This function saves all Q Tables to local drive after final training
        episode for analysis.
        
        '''
        
        dir_path = "./performance_log/" + timestamp + "/q_tables/"
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)        
        
        # Save all Q Tables
        q_tables = self.get_q_tables()[0]
        
        for idx, table in enumerate(q_tables): 
            
            table.to_csv(dir_path + "a" + str(idx) + "_q_table.csv")
        
        # Save the Consolidated Q Table
        self.knowledge_repo.to_csv(dir_path + "knowledge_repo.csv")
        

'''
-------------------------------------------------------------------------------
'''
        
class agent():
    
    '''
    this is a class object for the RL agent, which include learning
    and decisioning
    
    '''
    
    def __init__(self, name, action_space, epsilon, 
                 lr, gamma, current_stock, collaboration, trained_table = None):
        
        self.name = "a" + str(name)
        self.actions = action_space
        self.reward = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.current_stock = current_stock
        self.collaboration = collaboration
        
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

        
    def choose_action(self, s, knowledge_repo):
        
        '''
        This funciton choose an action based on Q Table. It also does 
        validation to ensure stock will not be negative after moving bikes.
        Input: 
            - s: current bike stock
            - actions: a mutable list from thread management to collect action
                        across multiple threads
            - idx: index to keep track of action per agent across multiple threads
        
        '''
        
        # Check if State Already Exist in Q-Table
        self.check_state_exist(s)
        self.current_stock = s
         
        if np.random.uniform() < self.epsilon:
            
            if self.collaboration:

                valid_state_action = self.q_table.loc[s, :]
                unique_value = valid_state_action.nunique()
                
                if unique_value > 1: 
                
                    try:
                        
                        state_actions = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                        action = state_actions.idxmax()
                    
                    except:
                        
                        action = 0
                    
                else:
                    
                    try:
    
                        valid_state_action = knowledge_repo.loc[s, :]
                        state_actions = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                        action = state_actions.idxmax()
                        
                    except:
                        
                        action = 0
                        
            else:
                
                # No Collaboration

                valid_state_action = self.q_table.loc[s, :]               
 
                try:
                    
                    state_actions = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                    action = state_actions.idxmax()
                
                except:
                    action = 0
                                              
        else:
            
            # randomly choose an action to explore
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
        
        '''
        This function creates a row for any new states the agent has not seen
        before and fill reward value with zeros.
        
        '''
        
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
        
    