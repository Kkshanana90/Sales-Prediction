#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import pickle
import time
import easygui
import xgboost as xgb


# ## Open Data and Join Data

# In[4]:


print("Please choose the TEST file in the dialogue (should be a CSV file - see ReadMe)")
filename = easygui.fileopenbox()
#filename = "data/train.csv"
data_train = pd.read_csv(filename)
data_store = pd.read_csv("data/store_AnSeb.csv")
dt = data_train.merge(data_store, left_on='Store', right_on='Store', how="left")


# In[ ]:





# ## Clean Data

# In[5]:


#Drop the Customers because we will not have it in the Future
dt = dt.drop("Customers", axis=1)


# In[6]:


#Date Variable format
dt['Date'] = pd.DatetimeIndex(dt['Date']) 


# ## Get rid of missing Stores (they miss too many features)

# In[7]:


#at this get rid of missing store
dt = dt.dropna( how='any', subset=["Store"])
#also change to INTEGER
dt.loc[:, "Store"] = dt.loc[:, "Store"].astype(int)


# # Feature Engineering

# In[8]:


dt.Open.fillna(1, inplace=True)

#Make a variable ReOPENING that takes the value 1 if a shop opens after 5 days of absence
dt['Open_5']  = dt.groupby('Store')['Open'].transform(lambda x: x.rolling(5,  min_periods=5).mean())
dt['Open_last5'] = dt.groupby('Store')['Open_5'].shift(1)

dt['Reopening'] = (dt.Open_last5 == 0) & (dt.Open == 1)


# In[9]:


#make one-hot-encoding for the StateHoliday variable
#a = public holiday, b = Easter holiday, c = Christmas, 0 = None
dt_eng = dt
dt_eng.loc[: , "PublicHoliday"] = dt.loc[:, "StateHoliday"]=="a" 
dt_eng.loc[: , "Easter"] = dt.loc[:, "StateHoliday"]=="b" 
dt_eng.loc[: , "Christmas"] = dt.loc[:, "StateHoliday"]=="c" 


# In[10]:


#StoreType One-Hot encoding
dummies = pd.get_dummies(dt.loc[:, "StoreType"], prefix="storetype", prefix_sep='_')
dt_eng = pd.concat([dt_eng, dummies], axis=1)


# In[11]:


#Assortment
dummies = pd.get_dummies(dt_eng.loc[:, "Assortment"], prefix="assort", prefix_sep='_')
dt_eng = pd.concat([dt_eng, dummies], axis=1)


# In[12]:


dt_eng.loc[:, "logDistance"] = np.log(dt_eng.loc[:, "CompetitionDistance"])


# ## Drop the Sales = 0 (after feature engineering)

# In[13]:


dt_eng = dt_eng.dropna( how='any', subset=['Sales'])
sales_zeros=(dt_eng["Sales"] == 0)
dt_eng = dt_eng.loc[~sales_zeros, :]


# In[14]:


#impute the School Holiday
dt_eng["SchoolHoliday"] = dt_eng["SchoolHoliday"].fillna(0)
dt_eng['CompetitionDistance'] = dt_eng['CompetitionDistance'].fillna(dt_eng['CompetitionDistance'].mean())
dt_eng['logDistance'] = dt_eng['logDistance'].fillna(dt_eng['logDistance'].mean())


# ### time series stuff

# In[15]:


##Add Monthly Fixed Effects
dt_eng['day'] = pd.DatetimeIndex(dt_eng['Date']).day
dt_eng['month'] = pd.DatetimeIndex(dt_eng['Date']).month
dt_eng['week'] = pd.DatetimeIndex(dt_eng['Date']).week
dt_eng['year'] = pd.DatetimeIndex(dt_eng['Date']).year


# In[16]:


dummies = pd.get_dummies(dt_eng.loc[:, "month"], prefix="m", prefix_sep='_')
dt_eng = pd.concat([dt_eng, dummies], axis=1)


# In[17]:


dt_eng = dt_eng.dropna( how='any', subset=["DayOfWeek"])
dummies = pd.get_dummies(dt_eng.loc[:, "DayOfWeek"].astype(int), prefix="dow", prefix_sep='_')
dt_eng = pd.concat([dt_eng, dummies], axis=1)


# In[18]:


#Beginning of the month craze
dt_eng["monthstart"] = (dt_eng.day>=30) | ( dt_eng.day<=3)


# In[19]:


dt_eng["prstart"] = (dt_eng.Promo2SinceWeek <= dt_eng.week) & (dt_eng.Promo2SinceYear <= dt_eng.year)
dt_eng = pd.concat([dt_eng, dt_eng["PromoInterval"].str.split(',', expand=True)], axis=1)


# In[20]:


def mnames(s):
    if s=="Jan":
        return 1
    if s=="Feb":
        return 2
    if s=="Mar":
        return 3
    if s=="Apr":
        return 4
    if s=="May":
        return 5
    if s=="Jun":
        return 6
    if s=="Jul":
        return 7
    if s=="Aug":
        return 8
    if s=="Sept":
        return 9
    if s=="Oct":
        return 10
    if s=="Nov":
        return 11
    if s=="Dec":
        return 12


# In[21]:


dt_eng.loc[:, 'pr1'] = dt_eng.loc[:, 0].apply(lambda row : mnames(row))
dt_eng.loc[:, 'pr2'] = dt_eng.loc[:, 1].apply(lambda row : mnames(row))
dt_eng.loc[:, 'pr3'] = dt_eng.loc[:, 2].apply(lambda row : mnames(row))
dt_eng.loc[:, 'pr4'] = dt_eng.loc[:, 3].apply(lambda row : mnames(row))

dt_eng.loc[:, "themonth"] = (dt_eng['pr1'] == dt_eng["month"])|(dt_eng['pr2'] == dt_eng["month"])|(dt_eng['pr4'] == dt_eng["month"])|(dt_eng['pr3'] == dt_eng["month"])  
dt_eng["pr_campaign"] = (dt_eng['prstart']==True) & (dt_eng["themonth"] == True)


# In[22]:


#Make a Variable that counts the days since day 1
dt_eng['Date'] = pd.to_datetime(dt_eng['Date']) 
dt_eng['date_delta'] = (dt_eng['Date'] - dt_eng['Date'].min())  / np.timedelta64(1,'D')


# In[23]:


#Define 'City_Center' =1 if distance <800 & competition open since year <2004
dt_eng['City_center'] = (dt_eng['CompetitionDistance'] < 500) &(dt_eng['CompetitionOpenSinceYear'] < 2004)


# In[24]:


#checkoing the data, one can see that in uneven weeks the first 3 days are selling especially well, this is one way to make it
dt_eng['week1'] = ((dt_eng['week'] % 4) == 0) 
dt_eng['week3'] = ((dt_eng['week'] % 4) == 2)
dt_eng['week13'] = ((dt_eng['week'] % 4) == 0) | ((dt_eng['week'] % 4) == 2)
dt_eng['week13'].value_counts()
dt_eng['firstdaysweek13'] = ((dt_eng['week13']) == True) & (dt_eng['DayOfWeek'] < 4 )


# In[25]:


#another way is to code the weekdays separately.... 
dt_eng["Fortnight_Days"] = dt_eng["DayOfWeek"]
dt_eng.Fortnight_Days.value_counts()
dt_eng.loc[dt_eng['week13']==True, "Fortnight_Days"] = dt_eng["Fortnight_Days"] * 2 
dt_eng.Fortnight_Days.value_counts()


# In[26]:


dic = {7: 1, 1: 2, 2:3, 3:4, 4:5, 5: 6 , 6: 7}
dt_eng['DayOfWeek_recode'] = dt_eng['DayOfWeek'].replace(dic)


# ## Final Feature selection

# In[37]:


features = [ 'SchoolHoliday', 'logDistance', 'prstart', 'Promo2', 'PublicHoliday', 'Easter', 'Christmas' 
            , 'Sales_avg_store', 'storetype_a', 'storetype_b', 'storetype_c', 'storetype_d', 'assort_a', 'assort_b', 'assort_c',
            "monthstart", "date_delta", "City_center", "firstdaysweek13", 
           "m_12", "Fortnight_Days", "DayOfWeek_recode",
           "dow_1", "dow_5", "dow_6",  "dow_2", "dow_3", "dow_4","dow_7" , "pr_campaign", "Reopening"]


# In[38]:


dt_eng_nomissing = dt_eng.dropna( how='any', subset=features)
X = dt_eng_nomissing.loc[:, features ]
y = dt_eng_nomissing.loc[:, 'Sales']


# ## Modelling

# ### Baseline

# In[39]:


def rmspe(preds, actuals):
    #preds = preds.reshape(-1)
    #actuals = actuals.reshape(-1)
    #assert preds.shape == actuals.shape
    return round(np.sqrt(np.mean(np.square((actuals-preds) / actuals))) * 100,4)


# In[41]:


m1 = pickle.load(open("model_AnSeb2.sav", 'rb'))

predictions = m1.predict(X)

print(f"Ansel and Sebastians model has an RMSPE of {rmspe(predictions, y)}")


# In[820]:


#Baseline
#y.loc[:, "Baseline"] = 5
#dt.loc[:, "Sales"].mean()
#y.loc[:, "new_row"] = 5
y = pd.DataFrame(y)
y.loc[:, "Baseline"] = y.loc[:, "Sales"].mean()
baseline_rmspe = rmspe(y.loc[:, "Baseline"], y.loc[:, "Sales"])
print(f"The Baseline (average) RMSPE is just {baseline_rmspe}")
time.sleep(5)
print(f".....looks like Ansel and Seb rock")

