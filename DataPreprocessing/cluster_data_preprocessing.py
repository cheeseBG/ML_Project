#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing for clustering

# ### Import module

# In[1]:


import pandas as pd
import numpy as np
import sys
import math
np.set_printoptions(threshold=sys.maxsize)
pd.options.display.float_format = '{:.2f}'.format

# ignore deep copy error
pd.set_option('mode.chained_assignment',  None)


# # Data Loading

# In[6]:


# C:\Users\JSW\Downloads\archive
# indicator dataset 
indi = pd.read_csv('../Data/indicators.csv')
# CN dataset
cn = pd.read_csv('../Data/Country.csv')


# In[7]:


d1 = indi
d2 = cn
data = d1.copy()
df_2 = d2.copy()


# # Selected indicators

# In[ ]:


# GDP NY.GDP.MKTP.CD
# C02 Emissions EN.ATM.CO2E.KT EN.ATM.CO2E.PC
# distibution of Phone IT.CEL.SETS
# Life expectancy SP.DYN.LE00.IN
# Fossil fuel Energy consumption EG.USE.COMM.FO.ZS
# Alternative Energy EG.USE.COMM.CL.ZS
# Arms Exports MS.MIL.XPRT.KD
# trade NE.TRD.GNFS.ZS 
# enegery use EG.USE.PCAP.KG.OE
# hospital beds(per 1,000 people) SH.MED.BEDS.ZS


# In[8]:


indi_list = ['GDP at market prices (current US$)','CO2 emissions (kt)','Mobile cellular subscriptions','Life expectancy at birth, total (years)','Fossil fuel energy consumption (% of total)','Alternative and nuclear energy (% of total energy use)','Arms exports (SIPRI trend indicator values)','Trade (% of GDP)','Energy use (kg of oil equivalent per capita)','Hospital beds (per 1,000 people)']
Country = data['CountryCode'].unique()
cols = ['CountryCode','GDP at market prices (current US$)','CO2 emissions (kt)','Mobile cellular subscriptions','Life expectancy at birth, total (years)','Fossil fuel energy consumption (% of total)','Alternative and nuclear energy (% of total energy use)','Arms exports (SIPRI trend indicator values)','Trade (% of GDP)','Energy use (kg of oil equivalent per capita)','Hospital beds (per 1,000 people)']


# # Create Dataset with merging

# In[9]:


# 처음 dataframe만들 때만 사용
# df : indicators.csv
# indicator : 사용할 indicator
# year : 선택할 year
def make_df(df,indicator,year):
    df_1 = df[df['Year'] == year]
    df_1 = df_1[df_1['IndicatorName'] == indicator]
    df_1 = df_1[['CountryCode','Value']]
    return df_1


# In[10]:


# data : indicators.csv
# df : merge하기전 dataframe
# indicator : 새로 merge하고싶은 indicator
# year : mearge 하기 위한 year 선택
# df_2 : 새로 merge하기위해 만들어진 dataframe
def data_merge(df,indicator,year):
    df_2 = data[data['Year'] == year]
    df_2 = df_2[df_2['IndicatorName'] == indicator]
    df_2 = df_2[['CountryCode','Value']]
    df_merge = pd.merge(df, df_2, on='CountryCode', how='outer')
    return df_merge


# # Dataset with 1960, 2010 year

# In[11]:


# Merge recursive function
def recur_data(data,year):
    for i in range(len(indi_list)):
        if i == 9:
            break
        data = data_merge(data,indi_list[i+1],year)
    return data


# In[12]:


df1 = make_df(data,indi_list[0],1960)
merge_last = recur_data(df1,1960)
merge_last.columns = cols
df_1960 = merge_last.copy()


# In[13]:


df1 = make_df(data,indi_list[0],2010)
merge_last = recur_data(df1,2010)
merge_last.columns = cols
df_2010 = merge_last.copy()


# In[14]:


df_1960


# In[91]:


df_2010


# In[ ]:


# There are too many 0(null) values


# # Check th Null value and fill null value with 3 criterions(Region, IncomeGroup, Currecy-Unit)

# In[15]:


# Check the Null value
def check_isnan_list(df,col):
    null_list = []
    for i in range(len(df)):
        if math.isnan(df[col][i]):
            null_list.append(i)
        else:
            continue
            
    return null_list

# Check the Null value
def check_isnan_dict(df,col):
    null_list = []
    for i in range(len(df)):
        if math.isnan(df[col][i]):
            null_list.append(i)
        else:
            continue
            
    null_dict = dict()
    for country in df['CountryCode'][null_list]:
        for index in range(len(df_2)):
            if df_2['CountryCode'][index] == country:
                region = df_2['Region'][index]
                null_dict[country] = region
    return null_dict


# In[16]:


# df1 : Indicator.csv , df2 : Country.csv
def Check_country_IncomeGroup(df1,df2,group,indicator,year):
    country_group = df2[df2['IncomeGroup'] == group]['CountryCode']
    data_group = df1[df1['CountryCode'].isin(country_group)]
    return data_group[(data_group['IndicatorName'] == indicator) & (data_group['Year'] == year)]

# df1 : Indicator.csv , df2 : Country.csv
def Check_country_region(df1,df2,region,indicator,year):
    country_region = df2[df2['Region'] == region]['CountryCode']
    data_region = df1[df1['CountryCode'].isin(country_region)]
    return data_region[(data_region['IndicatorName'] == indicator) & (data_region['Year'] == year)]

# df1 : Indicator.csv , df2 : Country.csv
def Check_CurrencyUnit(df1,df2,currencyunit,indicator,year):
    unit = df2[df2['CurrencyUnit'] == currencyunit]['CountryCode']
    data_unit = df1[df1['CountryCode'].isin(unit)]
    return data_unit[(data_unit['IndicatorName'] == indicator) & (data_unit['Year'] == year)]


# In[17]:


# Check the Null value
# df : merge가 끝난 dataframe
# col : null값을 채우기 위한 indicator value
# year : 해당 데이터셋의 year
# method : null값을 채우기 위한 기준
def change_null_value(df,col,year,method):
    null_list = []
    for i in range(len(df)):
        if math.isnan(df[col][i]):
            null_list.append(i)
        else:
            continue
            
    null_dict = dict()
    if method =='Region':
        for country in df['CountryCode'][null_list]:
            for index in range(len(df_2)):
                if df_2['CountryCode'][index] == country:
                    region = df_2[method][index]
                    null_dict[country] = region
        keys = list(null_dict.keys())        
        for key,null_value in zip(keys,null_list):
            new_value_data = Check_country_region(data,df_2,null_dict[key],col,year)
            new_value = np.average(new_value_data['Value'])
            df[col][null_value] = new_value
        return df
    
    elif method =='IncomeGroup':
        for country in df['CountryCode'][null_list]:
            for index in range(len(df_2)):
                if df_2['CountryCode'][index] == country:
                    group = df_2[method][index]
                    null_dict[country] = group
        keys = list(null_dict.keys())        
        for key,null_value in zip(keys,null_list):
            new_value_data = Check_country_IncomeGroup(data,df_2,null_dict[key],col,year)
            new_value = np.average(new_value_data['Value'])
            df[col][null_value] = new_value       
        return df
    
    elif method == 'CurrencyUnit':
        for country in df['CountryCode'][null_list]:
            for index in range(len(df_2)):
                if df_2['CountryCode'][index] == country:
                    unit = df_2[method][index]
                    null_dict[country] = unit
        keys = list(null_dict.keys())        
        for key,null_value in zip(keys,null_list):
            new_value_data = Check_CurrencyUnit(data,df_2,null_dict[key],col,year)
            new_value = np.average(new_value_data['Value'])
            df[col][null_value] = new_value       
        return df        


# In[18]:


print('As of 1960')
for i in range(len(indi_list)):
    nan_percent = len(check_isnan_list(df_1960, indi_list[i]))/237
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,indi_list[i],nan_percent))


# In[19]:


print('As of 2010')
for i in range(len(indi_list)):
    nan_percent = len(check_isnan_list(df_2010, indi_list[i]))/243
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,indi_list[i],nan_percent))


# # Drop indicators

# In[20]:


drop_indi = [4,5,6,7,8]


# # Fill null value with average of regions

# In[55]:


df_region_1960 = df_1960.copy()
df_region_2010 = df_2010.copy()


# In[56]:


df_region_1960.drop([indi_list[4],indi_list[5],indi_list[6],indi_list[7],indi_list[8]], axis='columns', inplace=True)
df_region_2010.drop([indi_list[4],indi_list[5],indi_list[6],indi_list[7],indi_list[8]], axis='columns', inplace=True)


# In[57]:


df_region_1960 = change_null_value(df_region_1960,indi_list[0],1960,'Region')
df_region_1960 = change_null_value(df_region_1960,indi_list[1],1960,'Region')
df_region_1960 = change_null_value(df_region_1960,indi_list[2],1960,'Region')
df_region_1960 = change_null_value(df_region_1960,indi_list[3],1960,'Region')
df_region_1960 = change_null_value(df_region_1960,indi_list[9],1960,'Region')


# In[24]:


df_region_2010 = change_null_value(df_region_2010,indi_list[0],1960,'Region')
df_region_2010 = change_null_value(df_region_2010,indi_list[1],1960,'Region')
df_region_2010 = change_null_value(df_region_2010,indi_list[2],1960,'Region')
df_region_2010 = change_null_value(df_region_2010,indi_list[3],1960,'Region')
df_region_2010 = change_null_value(df_region_2010,indi_list[9],1960,'Region')


# In[58]:


print('As of 1960')
for i in range(len(df_region_1960.columns)-1):
    nan_percent = len(check_isnan_list(df_region_1960, df_region_1960.columns[i+1]))/237
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,df_region_1960.columns[i+1],nan_percent))


# In[26]:


print('As of 2010')
for i in range(len(df_region_2010.columns)-1):
    nan_percent = len(check_isnan_list(df_region_2010, df_region_2010.columns[i+1]))/243
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,df_region_2010.columns[i+1],nan_percent))


# # Fill null value with average of IncomeGroup

# In[27]:


df_income_1960 = df_1960.copy()
df_income_2010 = df_2010.copy()
df_income_1960.drop([indi_list[4],indi_list[5],indi_list[6],indi_list[7],indi_list[8]], axis='columns', inplace=True)
df_income_2010.drop([indi_list[4],indi_list[5],indi_list[6],indi_list[7],indi_list[8]], axis='columns', inplace=True)


# In[28]:


df_income_1960 = change_null_value(df_income_1960,indi_list[0],1960,'IncomeGroup')
df_income_1960 = change_null_value(df_income_1960,indi_list[1],1960,'IncomeGroup')
df_income_1960 = change_null_value(df_income_1960,indi_list[2],1960,'IncomeGroup')
df_income_1960 = change_null_value(df_income_1960,indi_list[3],1960,'IncomeGroup')
df_income_1960 = change_null_value(df_income_1960,indi_list[9],1960,'IncomeGroup')


# In[29]:


df_income_2010 = change_null_value(df_income_2010,indi_list[0],1960,'IncomeGroup')
df_income_2010 = change_null_value(df_income_2010,indi_list[1],1960,'IncomeGroup')
df_income_2010 = change_null_value(df_income_2010,indi_list[2],1960,'IncomeGroup')
df_income_2010 = change_null_value(df_income_2010,indi_list[3],1960,'IncomeGroup')
df_income_2010 = change_null_value(df_income_2010,indi_list[9],1960,'IncomeGroup')


# In[49]:


print('As of 1960')
for i in range(len(df_income_1960.columns)-1):
    nan_percent = len(check_isnan_list(df_income_1960, df_income_1960.columns[i+1]))/237
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,df_income_1960.columns[i+1],nan_percent))


# In[31]:


print('As of 2010')
for i in range(len(df_income_2010.columns)-1):
    nan_percent = len(check_isnan_list(df_income_2010, df_income_2010.columns[i+1]))/243
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,df_income_2010.columns[i+1],nan_percent))


# # Fill null value with average of CurrencyUnit

# In[65]:


df_unit_1960 = df_1960.copy()
df_unit_2010 = df_2010.copy()
df_unit_1960.drop([indi_list[4],indi_list[5],indi_list[6],indi_list[7],indi_list[8]], axis='columns', inplace=True)
df_unit_2010.drop([indi_list[4],indi_list[5],indi_list[6],indi_list[7],indi_list[8]], axis='columns', inplace=True)


# In[66]:


df_unit_1960 = change_null_value(df_unit_1960,indi_list[0],1960,'CurrencyUnit')
df_unit_1960 = change_null_value(df_unit_1960,indi_list[1],1960,'CurrencyUnit')
df_unit_1960 = change_null_value(df_unit_1960,indi_list[2],1960,'CurrencyUnit')
df_unit_1960 = change_null_value(df_unit_1960,indi_list[3],1960,'CurrencyUnit')
df_unit_1960 = change_null_value(df_unit_1960,indi_list[9],1960,'CurrencyUnit')


# In[67]:


df_unit_2010 = change_null_value(df_unit_2010,indi_list[0],1960,'CurrencyUnit')
df_unit_2010 = change_null_value(df_unit_2010,indi_list[1],1960,'CurrencyUnit')
df_unit_2010 = change_null_value(df_unit_2010,indi_list[2],1960,'CurrencyUnit')
df_unit_2010 = change_null_value(df_unit_2010,indi_list[3],1960,'CurrencyUnit')
df_unit_2010 = change_null_value(df_unit_2010,indi_list[9],1960,'CurrencyUnit')


# In[48]:


print('As of 1960')
for i in range(len(df_unit_1960.columns)-1):
    nan_percent = len(check_isnan_list(df_unit_1960, df_unit_1960.columns[i+1]))/237
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,df_unit_1960.columns[i+1],nan_percent))


# In[47]:


print('As of 2010')
for i in range(len(df_unit_2010.columns)-1):
    nan_percent = len(check_isnan_list(df_unit_2010, df_unit_2010.columns[i+1]))/243
    print("{}.{}'s nan value % is {:.2f}%".format(i+1,df_unit_2010.columns[i+1],nan_percent))


# In[68]:


df_region_1960 = df_region_1960.where(pd.notnull(df_region_1960), df_region_1960.median(), axis='columns')
df_region_2010 = df_region_2010.where(pd.notnull(df_region_2010), df_region_2010.median(), axis='columns')
df_income_1960 = df_income_1960.where(pd.notnull(df_income_1960), df_income_1960.median(), axis='columns')
df_income_2010 = df_income_2010.where(pd.notnull(df_income_2010), df_income_2010.median(), axis='columns')
df_unit_1960 = df_unit_1960.where(pd.notnull(df_unit_1960), df_unit_1960.median(), axis='columns')
df_unit_2010 = df_unit_2010.where(pd.notnull(df_unit_2010), df_unit_2010.median(), axis='columns')

# 중간저장
df_region_1960.to_csv('../Data/sample/1960r.csv')
df_region_2010.to_csv('../Data/sample/2010r.csv')
df_income_1960.to_csv('../Data/sample/1960i.csv')
df_income_2010.to_csv('../Data/sample/2010i.csv')
df_unit_1960.to_csv('../Data/sample/1960u.csv')
df_unit_2010.to_csv('../Data/sample/2010u.csv')




