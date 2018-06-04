#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:49:14 2018

@author: Ian Xiao
"""

import numpy as np
import pandas as pd
import json

class env():
    
    def __init__(self, num_stations, mode):
        
        print("Creating a network of {} stations".format(num_stations))
        
        self.num_stations = num_stations
        self.mode = mode
        self.seed = np.random.random_integers(0, 10)
        self.num_hours = 23
        self.current_hour = 0
        self.stations_list = self.station_generator()
        
        self.get_station_info()
                        
    def station_generator(self):
        
        station_list = []
        
        for i in range(self.num_stations):
            station_list.append(station(i, 20))
            print("initiated station {}".format(i))
            
        return station_list
    
        
    def get_station_info(self):
        
        for station in self.stations_list:
            
            print("{}: {} bikes".format(station.get_name(), station.get_stock()))

class station():
    
    def __init__(self, name, starting_stock):
        self.name = "s" + str(name)
        self.starting_stock = starting_stock
        self.current_stock = starting_stock
        self.stock_flag = False
        self.threshold = 50
    
    def update_stock(self, stock_change):
        
        '''
        this function updates the current bike stock based on RL action and 
        simulated system dynamic
        '''
        self.current_stock = self.current_stock + stock_change
    
    def get_stock(self):
        
        '''
        this function prints hourly stock of the station
        '''
        
        return self.current_stock
        
    def get_name(self):
        
        '''
        this function print the station name
        '''
        
        return self.name
        
    def stock_check(self):
        
        '''
        this function flags if the stock is over or under the limits
        '''
        
        if self.current_stock > self.threshold or self.current_stock < 0:
            self.stock_flag = True