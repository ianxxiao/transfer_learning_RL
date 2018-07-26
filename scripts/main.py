#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:52:23 2018

@author: Ian Xiao
"""

from trainer import trainer

if __name__ == "__main__":
    
    # tODO: IMPLEMENT MULTIPROCESING
    
    action_space = [-50, -30, -20, -10, -3, -1, 0, 1, 3, 10, 20, 30, 50]
    num_stations = 3
    episode = 2000
    threshold = 3

    trainer = trainer(num_stations, action_space, episode, threshold, collaboration = True) 
    trainer.run(mode = "learn")    
    trainer.run(mode = "test")
    trainer.graph_performance(episode)
    
    del trainer
