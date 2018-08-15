#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 08:22:45 2018

@author: mrmrsxiao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_dir = "./performance_log/20180813235515061086/graphs/cost.csv"

cost = pd.read_csv(file_dir)
rolling_cost = pd.Series(cost.iloc[:, 1]).rolling(1000).mean()


fig = plt.figure(figsize = (10, 8))


x_axis = [x for x in range(len(rolling_cost))]
plt.plot(x_axis, rolling_cost)
plt.xlabel('Complete Network Success Incidence ordered by Episodes')
plt.ylabel("Rolling Cost (Total Reward - Success Reward)")
plt.title("Cost of Bike Moving for Complete Network Success")