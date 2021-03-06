#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:49:14 2018

@author: Ian Xiao
"""

import numpy as np
import pandas as pd

class env():
    
    def __init__(self, num_stations, init_stock, threshold):
        
        print("Creating a network of {} stations ...".format(num_stations))
        
        self.num_stations = num_stations
        self.init_stock = init_stock
        self.upper_trenshold = init_stock * threshold
        self.seed = np.random.random_integers(0, 10)
        self.num_hours = 23
        self.current_hour = 0
        self.hourly_flow_matrix = self.init_hourly_flow()
        self.stations_list = self.init_stations()
        self.rewards = np.zeros(self.num_stations)
        self.limit_flags = np.zeros(self.num_stations)
        self.day_end = False
        self.mode = "learn"
        
        self.success_flags = []
        
        for station in range(self.num_stations):
            self.success_flags.append(True)
        
    def init_stations(self):
        
        '''
        this function generates station class based on the specified number 
        of stations. 
        
        return: a list of station class (e.g. name, simulated hourly stock)
        
        '''
        
        station_list = []
        # create station class with pre-defined hourly flow
        
        for i in range(self.num_stations):
            
            if self.num_stations > 1:
            
                all_hourly_inflow = []
                all_hourly_outflow = []
                
                # get hourly inflow and outflow
                for hour in range(self.num_hours):
                    
                    this_hour_flow_matrix = self.hourly_flow_matrix[hour]
                    
                    this_hour_inflow = 0
                    this_hour_outfow = 0
                    
                    for idx in range(len(this_hour_flow_matrix)):
                        # row wise addition
                        this_hour_inflow += this_hour_flow_matrix[idx][i]
                        
                        # column wise addition
                        this_hour_outfow += this_hour_flow_matrix[i][idx]
                    
                    all_hourly_inflow.append(this_hour_inflow)
                    all_hourly_outflow.append(this_hour_outfow)            
                            
                # compute hourly netflow
                this_station_hourly_net_flow = np.array(all_hourly_inflow) - np.array(all_hourly_outflow)
                                
                station_list.append(station(i, self.init_stock, this_station_hourly_net_flow))
            
            else:
                # create station class

                station_list.append(station(i, self.init_stock, self.hourly_flow_matrix))
            
        return station_list
            

    def init_hourly_flow(self):
        
        '''
        This function calculate the hourly 
        
        '''
                
        hourly_flow = []
        
        for hour in range(self.num_hours):
            if self.num_stations != 1:
                bike_flow = np.random.random_integers(0, 20, (self.num_stations, 
                                                          self.num_stations))
            else: 
                bike_flow = int(np.random.random_integers(0, 20, (1)))        
            
            # replace diagonal with zeros
            if self.num_stations > 1:
                for idx in range(self.num_stations):
                    bike_flow[idx][idx] = 0
            
            hourly_flow.append(bike_flow)

        return hourly_flow

        
    def ping(self, actions):
        
        '''
        
        This function triggers the environment reaction to agents' actions
        (e.g. update bike stock, update clock, etc.)
        
        '''
        
        # Update Station Stocks
        # actions = [-3, -5, 1] as number of bikes move for the corresponding station
        
        for station in range(self.num_stations):
        
            self.stations_list[station].update_stock(self.current_hour, actions[station])
            
            self.rewards[station] = -0.5*np.abs(actions[station])
        
        # Update Env Variables
        if self.current_hour == 23:
            
            self.day_end = True
            
            for station in range(self.num_stations):
                if self.success_flags[station] == True:
                    self.rewards[station] += 50
                    
                    #TODO: ADD TEAM SUCCESS REWARD
                    
                else: 
                    self.rewards[station] += -50
            
        if self.current_hour != 23:
            self.update_hour()
            
            self.old_stocks = self.get_old_stocks() 
            self.new_stocks = self.get_new_stocks()
            
            # Penalize RL agent if stocks is over limits
            for station in range(self.num_stations):
                if self.new_stocks[station] > self.upper_trenshold or self.new_stocks[station] <= 0:
                    self.rewards[station] += -30
                    self.success_flags[station] = False
        
        return self.current_hour, self.old_stocks, self.new_stocks, self.rewards, self.day_end
        
            
    def update_hour(self):
        
        '''
        This function updates the environment clock.
        
        '''
        
        self.current_hour += 1
        
    
    def get_old_stocks(self):
        
        '''
        This function collects 
        
        '''
        
        # get old stocks from each station
        old_stocks = []
        
        for station in self.stations_list:
            
            old_stocks.append(station.get_old_stock())
            
        return old_stocks

        
    def get_new_stocks(self):
        
        # get new stocks from each station
        
        new_stocks = []
        
        for station in self.stations_list:
            
            new_stocks.append(station.get_new_stock())
                        
        return new_stocks

        
    def get_station_info(self):
        
        for station in self.stations_list:
            
            print("{}: {} bikes".format(station.get_name(), station.get_old_stock()))
            
    def eps_reset(self):
                
        self.current_hour = 0
        self.hourly_flow_matrix = self.init_hourly_flow()
        self.stations_list = self.init_stations()
        self.rewards = np.zeros(self.num_stations)
        self.limit_flags = np.zeros(self.num_stations)
        self.day_end = False
        
        self.success_flags = []
        
        for station in range(self.num_stations):
            self.success_flags.append(True)
        
    def cal_success_ratio(self):
        
        return sum(self.success_flags)*1.0 / len(self.success_flags)
        
        
'''

------------------------------------------------------------------------------


'''
            
class station():
    
    def __init__(self, name, starting_stock, hourly_net_flow):
        self.name = "s" + str(name)
        self.starting_stock = starting_stock
        self.current_stock = starting_stock
        self.stock_flag = False
        self.threshold = 50
        self.hourly_net_flow = hourly_net_flow
        self.bike_stock_sim = self.generate_stock()
        self.bike_stock = self.bike_stock_sim # to be reset to original copy every episode
        self.old_stock = self.bike_stock[0]
        self.new_stock = self.bike_stock[0]
        self.day_end = False
        self.bike_moved = 0
        
        
    def generate_stock(self):
        
        stock = [self.starting_stock]
        
        for i in range(1, 24):
            stock.append(stock[i-1] + self.hourly_net_flow[i-1])
        
        #print("{} - {}".format(self.name, stock))
            
        return stock
             
        
    def update_stock(self, current_hour, action):
        
        '''
        this function updates the current bike stock based on RL action
        '''
        
        # update bike stock based on RL agent action at t
        if current_hour < 23: 
            for hour in range(current_hour+1, len(self.bike_stock)):
                self.bike_stock[hour] += action
            
            # update snapshot information    
            self.bike_moved = action
            self.old_stock = self.bike_stock[current_hour]
            self.new_stock = self.bike_stock[current_hour+1]
            
            #print("{} - {}".format(self.name, self.bike_stock))
            
        else: 
            pass
    
        
    def get_old_stock(self):
        
        return self.old_stock
    
        
    def get_new_stock(self):
        
        return self.new_stock
        
        
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