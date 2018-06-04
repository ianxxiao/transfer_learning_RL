#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:52:23 2018

@author: mrmrsxiao
"""

from env import env


if __name__ == "__main__":

    env1 = env(3, "random")
    env1.ping([-10, -10, -10])
    env1.ping([-10, -10, -10])