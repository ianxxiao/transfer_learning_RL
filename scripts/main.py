#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:52:23 2018

@author: Ian Xiao
"""

from trainer import trainer

if __name__ == "__main__":
    
    action_space = [-10, -3, -1, 0]
    trainer = trainer(3, action_space, eps = 100)
    
    env1 = env(3, 50)
    agent_manager1 = agent_manager(3, [-10, -3, -1, 0], [50, 50, 50])
    
    actions = [0, 0, 0]

    for hour in range(0, 24):
        
        current_hour, old_stocks, new_stocks, rewards, day_end = env1.ping(actions)
        
        agent_manager1.batch_learn(old_stocks, actions, rewards, new_stocks, day_end)
        
        actions = agent_manager1.batch_choose_action(new_stocks)
        
        print("acionts: {}".format(actions))