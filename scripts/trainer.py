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
        self.init_stock = 50
        init_stock = []
        for idx in range(len(num_stations)):
            init_stock.append(self.init_stock)
        
        self.agent_manager = agent_manager(num_stations, action_space)
        
        if type(eps) == "int":
            self.eps = [eps]
            
        elif type(eps) == "list":
            self.eps = eps
        
    def start(self):
        
        print("start training session ...")