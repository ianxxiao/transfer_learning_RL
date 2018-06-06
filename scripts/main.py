#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:52:23 2018

@author: Ian Xiao
"""

from env import env
from rl_brain import agent_manager


if __name__ == "__main__":

    env1 = env(2, 50)
    agent_manager1 = agent_manager(2, [-10, -3, -1, 0], [50, 50])

    for hour in range(0, 24):
        
        current_hour, old_stocks, new_stocks, rewards, done, game_over = env1.ping([0, 0])
        
        agent_manager1.batch_learn(old_stocks, [0, 0], rewards, new_stocks, False)