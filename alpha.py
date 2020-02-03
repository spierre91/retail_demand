from wqpt import predict, fit, set_state
import pandas as pd 
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from pandas.core.indexing import IndexingError
import numpy as np
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
pd.options.mode.chained_assignment = None 
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)


def date_mapper(date_str):
    return datetime.strptime(date_str, '%m/%d/%Y')


class Alpha:
    def __init__(self, datafile="files/data.csv"):
        print("Initializing...")        
        self.df_scope = pd.read_csv(datafile)
        self.commands = pd.read_json("commands.json",  lines=True)
        Upc = []
        for i in list(self.commands['args'].values):
            Upc.append(i['Upc'])
        self.Upc = list(set(Upc))

        self.df = self.df_scope[[ 'AvgPctAcv', 'AvgPctAcvYago', 'Units', 'UnitsYago', 'Upc', 'base_price',
                                 'base_price_yago', 'discount_perc', 'discount_perc_yago', 'Flavor', 'Brand', 'SPPD', 'TimePeriodEndDate',
                                 'AvgPctAcvAnyDisplay', 'AvgPctAcvAnyFeature', 'AvgPctAcvFeatureAndDisplay', 
                                 'AvgPctAcvTpr']]       

        self.df['TimePeriodEndDate'] = pd.to_datetime(self.df['TimePeriodEndDate'], format='%m/%d/%Y')
        self.df.reset_index(inplace = True)
        self.START_DATE = self.df['TimePeriodEndDate'].loc[0]
        self.df['month_number'] = self.df['TimePeriodEndDate'].dt.month//1
        self.df['week'] =  self.df['TimePeriodEndDate'].dt.day//7
        self.df['year'] =  self.df['TimePeriodEndDate'].dt.year//1    
        self.df['weeks_since_start'] =  (self.df['TimePeriodEndDate']  - self.START_DATE).dt.days // 7    
        self.df['other_acv'] = (1 - self.df['AvgPctAcv'])
 
        self.df['Units'].fillna(1, inplace = True)
        self.df['base_price'].fillna(1, inplace = True)
        
        self.df['discount_perc'].fillna(0, inplace = True)
        self.df['list'] =  self.df['base_price']*(1 -  self.df['discount_perc'])
        self.df['list'].fillna(1, inplace = True)
        
        self.df['AvgPctAcvAnyDisplay'].fillna(0, inplace = True)
        self.df['AvgPctAcvAnyFeature'].fillna(0, inplace = True)
        
        self.df['AvgPctAcvFeatureAndDisplay'].fillna(0, inplace = True)        
        self.df['AvgPctAcvTpr'].fillna(0, inplace = True)
        
        self.df['pct_change_base'] = list(self.df['base_price'].pct_change())
        self.df['pct_change_base'].replace([np.inf, -np.inf], np.nan, inplace = True)
        self.df['pct_change_base'].fillna(0, inplace = True) 
        
        self.training_max_date = None
        self.models = {}
        self.orginial = self.df.copy(deep = True)
        self.orginial.dropna(inplace = True)
        self.df.dropna(inplace = True)
    def set_state(self, TimePeriodEndDate):
        print("Setting state...")        
        self.training_max_date =  date_mapper(TimePeriodEndDate)
        # dhe
        self.models.clear()
        # end dhe 
        
                
    def fit(self):
        print("Fitting...")
        feature_list = ['base_price', 'discount_perc', 'AvgPctAcv', 'AvgPctAcvYago', 'UnitsYago', 'base_price_yago', 
                         'discount_perc_yago', 'AvgPctAcvAnyDisplay', 'AvgPctAcvAnyFeature', 'AvgPctAcvFeatureAndDisplay', 
                         'AvgPctAcvTpr', 'list', 'other_acv', 'pct_change_base', 'month_number', 'week','year', 'weeks_since_start']        
        if self.training_max_date is None:
            raise Exception(
                'attempting to fit models before any data '
                'is made available')   
        mask = (self.df['TimePeriodEndDate'] <= datetime(self.training_max_date.year,  self.training_max_date.month, self.training_max_date.day))    
        reg = {}
        self.models['reg'] = dict()

        self.df = self.orginial
        self.df.dropna(inplace = True)
        try:
            X = np.array(self.df.loc[mask][feature_list])
            y = np.array(self.df.loc[mask]["SPPD"]) 
        except IndexingError as IE:
            print(IE)
            X = np.array(self.df[feature_list])
            y = np.array(self.df["SPPD"]) 
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        reg = RandomForestRegressor(random_state = 42)
        reg = reg.fit(X, y)
        self.models['reg_all'] = reg    
        for i in self.Upc:
            try:
                self.df.dropna(inplace = True)  
                self.df = self.orginial
                self.df = self.df[self.df['Upc'] == i]
                X = np.array(self.df.loc[mask][feature_list])
                y = np.array(self.df.loc[mask]["SPPD"]) 
                X = np.nan_to_num(X)
                y = np.nan_to_num(y)
                reg = RandomForestRegressor(random_state = 42)
                reg = reg.fit(X, y)             
                self.models['reg'][i] = reg  
                self.df = self.orginial                 
            except(ValueError):
                self.df.dropna(inplace = True) 
                self.models['reg'][i] = self.models['reg_all'] 

    def predict(self, TimePeriodEndDate, Upc, base_price, discount_perc, AvgPctAcv, AvgPctAcvAnyDisplay, 
                AvgPctAcvAnyFeature, AvgPctAcvFeatureAndDisplay, AvgPctAcvTpr):
        print("Predicting...")
        self.df.dropna(inplace = True) 
        DATE = date_mapper(TimePeriodEndDate)
        month_number = DATE.month//1
        week = DATE.day//7
        year = DATE.year//1
        weeks_since_start = (date_mapper(TimePeriodEndDate)  - self.START_DATE).days // 7     
        lp = base_price*(1-discount_perc)
        other_acv = (1 - AvgPctAcv)
        pct_change_base = self.df['pct_change_base'].mean()  
        AvgPctAcvYago = self.df['AvgPctAcvYago'].mean()         
        UnitsYago = self.df['UnitsYago'].mean()         
        base_price_yago= self.df['base_price_yago'].mean() 
        discount_perc_yago = self.df['discount_perc_yago'].mean() 
        input_list = [base_price, discount_perc, AvgPctAcv, AvgPctAcvYago, UnitsYago, 
                             base_price_yago, discount_perc_yago, AvgPctAcvAnyDisplay, AvgPctAcvAnyFeature, AvgPctAcvFeatureAndDisplay, 
                         AvgPctAcvTpr, lp, other_acv, pct_change_base, month_number, week, year, weeks_since_start]
        
        input_list = [0 if math.isnan(x) else x for x in input_list]
        try:
            result =  self.models['reg'][Upc].predict(np.array([input_list]).reshape(1,-1))   
        except KeyError as KE:
            print(KE)
            result =  self.models['reg_all'].predict(np.array([input_list]).reshape(1,-1)) 

        print("Result:", result[0])

        return result[0] if result[0] > 0 else 0
        
        
# Create alpha instance
model = Alpha()


@fit
def fit():
    model.fit()


@set_state
def set_state(TimePeriodEndDate):
    model.set_state(TimePeriodEndDate=TimePeriodEndDate)


@predict
def predict(TimePeriodEndDate, Upc, base_price, discount_perc, AvgPctAcv, AvgPctAcvAnyDisplay, AvgPctAcvAnyFeature, AvgPctAcvFeatureAndDisplay, AvgPctAcvTpr):
    return model.predict(TimePeriodEndDate=TimePeriodEndDate, Upc=Upc, base_price=base_price, discount_perc=discount_perc, AvgPctAcv=AvgPctAcv, AvgPctAcvAnyDisplay=AvgPctAcvAnyDisplay, AvgPctAcvAnyFeature=AvgPctAcvAnyFeature, AvgPctAcvFeatureAndDisplay=AvgPctAcvFeatureAndDisplay, AvgPctAcvTpr=AvgPctAcvTpr)

def main():
    print("Hi")


if __name__ == '__main__':
    main()