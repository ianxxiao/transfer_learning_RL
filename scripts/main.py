#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:52:23 2018

@author: Ian Xiao
"""

from trainer import trainer

if __name__ == "__main__":
    
    # Introduce Parallel Processing Scheme
    # Add Action-Value Transfer Learning Mechanics
    
    action_space = [-20, -10, -3, -1, 0, 1, 3, 10, 20]
    num_stations = 3
    episode = 50000

    trainer = trainer(num_stations, action_space, episode)
    trainer.run(mode = "learn")    
    trainer.run(mode = "test")
    trainer.graph_performance(episode)
    
    del trainer
