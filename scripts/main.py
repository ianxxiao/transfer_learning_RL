#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:52:23 2018

@author: Ian Xiao
"""

from env import env


if __name__ == "__main__":

    env1 = env(5, 50)
    env2 = env(7, 30)
    
    for hour in range(0, 24):
        env1.ping([-10, -10, -10, -10, -10])

    # reset 
    print("===============================")
    env1.reset()
    for hour in range(0, 24):
        env1.ping([-10, -10, -10, -10, -10])