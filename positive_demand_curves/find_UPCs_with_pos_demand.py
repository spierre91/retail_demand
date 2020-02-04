#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:25:45 2020

@author: sadrachpierre
"""

import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("files/data.csv")
commands = pd.read_json("commands.json",  lines=True)

df['TimePeriodEndDate'] = pd.to_datetime(df['TimePeriodEndDate'])
df['year'] = df['TimePeriodEndDate'].dt.year
df['base_price'] = df['base_price'].round(1)
df['discount_perc'] = df['discount_perc'].round(1)
df = df[['Upc', 'base_price', 'year', 'discount_perc', 'SPPD']]
df = df.groupby(['base_price', 'year', 'Upc'])['discount_perc'].value_counts()
df = df[df.values >= 5]
df.to_csv("upc_gt5_unique_discounts.csv")
full_upc_list = list(df.index.get_level_values('Upc'))

df_full = pd.read_csv("files/data.csv")
from sklearn.linear_model import LinearRegression 
models = {}
df = df_full[df_full['Upc'].isin(full_upc_list)]


df['discount_perc'] = df['discount_perc'].fillna(0)
df['SPPD'] = df['SPPD'].fillna(0)
rsquared = []
sku_list = []
import numpy as np 
rsquare = pd.read_csv("upc_dicsount_sppd_r2_gt08_update.csv")
length = []
for i in list(set(rsquare['upc'].values)):
    try:
        df_new = df[df['Upc'] == i]
        df_zero = df_new[df_new['discount_perc'] == 0]
        
        df_005 = df_new[df_new['discount_perc'] >= 0.05]
        df_005 = df_005[df_005['discount_perc'] < 0.1]
        
        
        df_01 = df_new[df_new['discount_perc'] >= 0.1]
        df_01 = df_01[df_01['discount_perc'] < 0.15]
        
        df_015 = df_new[df_new['discount_perc'] >= 0.15]
        df_015 = df_015[df_015['discount_perc'] < 0.2]
        
        df_02 = df_new[df_new['discount_perc'] >= 0.2]
        df_02 = df_02[df_02['discount_perc'] < 0.25]
        
        df_025 = df_new[df_new['discount_perc'] >= 0.25]
        df_025 = df_025[df_025['discount_perc'] < 0.3]
        
        
        df_03 = df_new[df_new['discount_perc'] >= 0.3]
        df_03 = df_03[df_03['discount_perc'] < 0.35]
        
        df_035 = df_new[df_new['discount_perc'] >= 0.35]
        df_035 = df_035[df_035['discount_perc'] < 0.4]
            
        
        df_04 = df_new[df_new['discount_perc'] >= 0.4]
        df_04 = df_04[df_04['discount_perc'] < 0.45]
        
        
        df_045 = df_new[df_new['discount_perc'] >= 0.45]
        df_045 = df_045[df_045['discount_perc'] < 0.5]
        
        
        df_05 = df_new[df_new['discount_perc'] >= 0.5]
        df_05 = df_05[df_05['discount_perc'] < 0.55]
        
        df_055 = df_new[df_new['discount_perc'] >= 0.55]
        df_055 = df_055[df_055['discount_perc'] < 0.6]
        
        
        df_06 = df_new[df_new['discount_perc'] >= 0.6]
        df_06 = df_06[df_06['discount_perc'] < 0.65]
        
        
        df_065 = df_new[df_new['discount_perc'] >= 0.65]
        df_065 = df_065[df_065['discount_perc'] < 0.7]
        
        
        df_07 = df_new[df_new['discount_perc'] >= 0.7]
        df_07 = df_07[df_07['discount_perc'] < 0.75]
        
        df_075 = df_new[df_new['discount_perc'] >= 0.75]
        df_075 = df_075[df_075['discount_perc'] < 0.8]
        
        
        df_08 = df_new[df_new['discount_perc'] >= 0.8]
        df_08 = df_08[df_08['discount_perc'] < 0.85]
    
    
        actual_list = [df_zero['SPPD'].mean(), df_005['SPPD'].mean(), df_01['SPPD'].mean(), df_015['SPPD'].mean(),
                       df_02['SPPD'].mean(), df_025['SPPD'].mean(), df_03['SPPD'].mean(),
                       df_035['SPPD'].mean(), df_04['SPPD'].mean(), df_045['SPPD'].mean(), 
                       df_05['SPPD'].mean(), df_055['SPPD'].mean(), df_06['SPPD'].mean(), 
                       df_065['SPPD'].mean(), df_07['SPPD'].mean(), df_075['SPPD'].mean(),
                       df_08['SPPD'].mean()]
        
        discount_list = [ 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
        df_demand = pd.DataFrame({"discount": discount_list,
                                "actual_avg_sppd_demand": actual_list})
        df_demand.dropna(inplace=True)
        X = np.array(df_demand["discount"].values).reshape(-1, 1)
        y = np.array(df_demand["actual_avg_sppd_demand"].values)
    
        reg = LinearRegression()
        reg.fit(X,y)
        plt.scatter(X,y)
        plt.show()
        print(reg.score(X,y))
        sku_list.append(i)
        rsquared.append(reg.score(X,y))
        length.append(len(X))
        print(len(X))
    except(ValueError):
        sku_list.append(i)
        rsquared.append(np.nan) 
        length.append(len(X))
        print(len(X))
#        
#good_upc = pd.DataFrame({'upc': sku_list, 'rsquared': rsquared, 'length': length})  
#good_upc.to_csv("full_upc_dicsount_sppd_r2.csv")    
#print(good_upc.head())
        
        
#good_upc = good_upc[good_upc['rsquared']> 0.8]    
#good_upc = good_upc[good_upc['length']> 5]    
#good_upc.to_csv("upc_dicsount_sppd_r2_gt08_update.csv")  
#print(good_upc.head(100))
        
#print(len(discount_list))
#print(len(actual_list))
#
#df_demand = pd.DataFrame({"discount": discount_list,
#                            "actual_avg_sppd_demand": actual_list})
#print(df_demand.head())
##df_demand = df_demand[df_demand["discount"]<= 0.4]
#
#plt.scatter(df_demand["discount"], df_demand["actual_avg_sppd_demand"])
##plt.scatter(df_demand["discount"], df_demand["predict"])
#
#plt.xlabel("Discount")
#plt.ylabel("AVG_SPPD")
#plt.title(title)
