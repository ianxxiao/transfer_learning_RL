#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:36:11 2018

@author: Ian Xiao
"""

from env import env
from rl_brain import agent_manager

class trainer():
    
    
    def __init__(self, num_stations, action_space, eps):
        
        self.env = env(num_stations, 50)
        self.init_stocks = []
        self.current_stocks = self.init_stocks
        self.num_stations = num_stations
        self.action_space = action_space
                
        for idx in range(num_stations):
            self.init_stocks.append(50)
        
        self.agent_manager = agent_manager(self.num_stations, 
                                           self.action_space, self.init_stocks)
        
        if type(eps) == "int":
            self.eps = [eps]
            print(self.eps)
            
        elif type(eps) == "list":
            episode_list = [eps for eps in range(eps[0], eps[1]+1, eps[2])]
            self.eps = episode_list
            print(self.eps)
        
    def start(self):
        
        print("==========================")
        print("start training session ...")
        
        for hour in range(0, 24):
            
            actions = self.agent_manager.batch_choose_action(self.current_stocks)
            
            current_hour, old_stocks, new_stocks, rewards, day_end = self.env.ping(actions)
            
            self.agent_manager.batch_learn(old_stocks, actions, rewards, new_stocks, day_end)
            
            self.current_stocks = new_stocks
            
