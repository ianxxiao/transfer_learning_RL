#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:52:23 2018

@author: Ian Xiao
"""

from trainer import trainer

if __name__ == "__main__":
    
    action_space = [-10, -3, -1, 0, 1, 3, 10]
    trainer = trainer(3, action_space, episode = 10000)
    trainer.start()
    
    del trainer
