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
indi = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\indicators.csv')
# CN dataset
cn = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\Country.csv')


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


print('1960년 기준')
for i in range(len(indi_list)):
    nan_percent = len(check_isnan_list(df_1960, indi_list[i]))/237
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,indi_list[i],nan_percent))


# In[19]:


print('2010년 기준')
for i in range(len(indi_list)):
    nan_percent = len(check_isnan_list(df_2010, indi_list[i]))/243
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,indi_list[i],nan_percent))


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


print('1960년 기준')
for i in range(len(df_region_1960.columns)-1):
    nan_percent = len(check_isnan_list(df_region_1960, df_region_1960.columns[i+1]))/237
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,df_region_1960.columns[i+1],nan_percent))


# In[26]:


print('2010년 기준')
for i in range(len(df_region_2010.columns)-1):
    nan_percent = len(check_isnan_list(df_region_2010, df_region_2010.columns[i+1]))/243
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,df_region_2010.columns[i+1],nan_percent))


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


print('1960년 기준')
for i in range(len(df_income_1960.columns)-1):
    nan_percent = len(check_isnan_list(df_income_1960, df_income_1960.columns[i+1]))/237
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,df_income_1960.columns[i+1],nan_percent))


# In[31]:


print('2010년 기준')
for i in range(len(df_income_2010.columns)-1):
    nan_percent = len(check_isnan_list(df_income_2010, df_income_2010.columns[i+1]))/243
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,df_income_2010.columns[i+1],nan_percent))


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


print('1960년 기준')
for i in range(len(df_unit_1960.columns)-1):
    nan_percent = len(check_isnan_list(df_unit_1960, df_unit_1960.columns[i+1]))/237
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,df_unit_1960.columns[i+1],nan_percent))


# In[47]:


print('2010년 기준')
for i in range(len(df_unit_2010.columns)-1):
    nan_percent = len(check_isnan_list(df_unit_2010, df_unit_2010.columns[i+1]))/243
    print('{}.{}의 nan value % 는 {:.2f}% 입니다'.format(i+1,df_unit_2010.columns[i+1],nan_percent))


# In[68]:


df_region_1960 = df_region_1960.where(pd.notnull(df_region_1960), df_region_1960.median(), axis='columns')
df_region_2010 = df_region_2010.where(pd.notnull(df_region_2010), df_region_2010.median(), axis='columns')
df_income_1960 = df_income_1960.where(pd.notnull(df_income_1960), df_income_1960.median(), axis='columns')
df_income_2010 = df_income_2010.where(pd.notnull(df_income_2010), df_income_2010.median(), axis='columns')
df_unit_1960 = df_unit_1960.where(pd.notnull(df_unit_1960), df_unit_1960.median(), axis='columns')
df_unit_2010 = df_unit_2010.where(pd.notnull(df_unit_2010), df_unit_2010.median(), axis='columns')


# ## Scaling & PCA

# ### import module

# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


# In[38]:


df_region_1960.describe()


# In[39]:


sd = StandardScaler()


# In[40]:


new_cols = df_region_1960.columns


# In[41]:


new_cols[1:6]


# In[69]:


sd_region_1960 = sd.fit_transform(df_region_1960[new_cols[1:6]])
sd_region_2010 = sd.fit_transform(df_region_2010[new_cols[1:6]])
sd_income_1960 = sd.fit_transform(df_income_1960[new_cols[1:6]])
sd_income_2010 = sd.fit_transform(df_income_2010[new_cols[1:6]])
sd_unit_1960 = sd.fit_transform(df_unit_1960[new_cols[1:6]])
sd_unit_2010 = sd.fit_transform(df_unit_2010[new_cols[1:6]])


# In[126]:


# 중간저장
df_region_1960.to_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\1960r.csv')
df_region_2010.to_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\2010r.csv')
df_income_1960.to_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\1960i.csv')
df_income_2010.to_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\2010i.csv')
df_unit_1960.to_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\1960u.csv')
df_unit_2010.to_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\2010u.csv')


# In[3]:


# 불러오기
df_region_1960 = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\1960r.csv')
df_region_2010 = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\2010r.csv')
df_income_1960 = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\1960i.csv')
df_income_2010 = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\2010i.csv')
df_unit_1960 = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\1960u.csv')
df_unit_2010 = pd.read_csv('C:\\Users\\JSW\\Downloads\\archive\\sample\\2010u.csv')


# In[43]:


df_income_1960


# In[125]:


sd_region_1960


# In[125]:


sd_region_1960


# In[70]:


pca = PCA(n_components=2)
X_principal = pca.fit_transform(sd_region_1960)
X_principal_region_1960 = pd.DataFrame(X_principal)
X_principal_region_1960.columns = ['P1_1960r','P2_1960r']

X_principal = pca.fit_transform(sd_region_2010)
X_principal_region_2010 = pd.DataFrame(X_principal)
X_principal_region_2010.columns = ['P1_2010r','P2_2010r']

X_principal = pca.fit_transform(sd_income_1960)
X_principal_income_1960 = pd.DataFrame(X_principal)
X_principal_income_1960.columns = ['P1_1960i','P2_1960i']

X_principal = pca.fit_transform(sd_region_2010)
X_principal_income_2010 = pd.DataFrame(X_principal)
X_principal_income_2010.columns = ['P1_2010i','P2_2010i']

X_principal = pca.fit_transform(sd_unit_1960)
X_principal_unit_1960 = pd.DataFrame(X_principal)
X_principal_unit_1960.columns = ['P1_1960u','P2_1960u']

X_principal = pca.fit_transform(sd_unit_2010)
X_principal_unit_2010 = pd.DataFrame(X_principal)
X_principal_unit_2010.columns = ['P1_2010u','P2_2010u']


# In[71]:


region_pca_df = [X_principal_region_1960, X_principal_region_2010]
income_pca_df = [X_principal_income_1960, X_principal_income_2010]
unit_pca_df = [X_principal_unit_1960, X_principal_unit_2010]


# # Clustering

# ### Import module

# In[72]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import mixture
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist


# ## DBSCAN

# In[74]:


# hpyer parameter setting
eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
min_samples = [5,10,20,50,100]
algorithm_euc = ['ball_tree','kd_tree','brute']
algorithm_ha = ['ball_tree', 'brute']
distance = ['euclidean', 'hamming'] 


# In[82]:


# Euclidean
cnt = 0
index = 0
result = []
for d in region_pca_df:
    result_df = []
    for e in eps:
        for m in min_samples:
            for a in algorithm_euc:
                db = DBSCAN(metric = distance[0], eps=e, min_samples=m, algorithm=a).fit(d)
                labels = db.labels_
                plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
                plt.title((e,'eps with', m,'min_samples',a,'Algorithm & Euclidean'))
                plt.show()
    cnt += 1


# In[ ]:


# Euclidean
cnt = 0
index = 0
result = []
for d in income_pca_df:
    result_df = []
    for e in eps:
        for m in min_samples:
            for a in algorithm_euc:
                db = DBSCAN(metric = distance[0], eps=e, min_samples=m, algorithm=a).fit(d)
                labels = db.labels_
                plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
                plt.title((e,'eps with', m,'min_samples',a,'Algorithm & Euclidean'))
                plt.show()
    cnt += 1


# In[ ]:


# Euclidean
cnt = 0
index = 0
result = []
for d in unit_pca_df:
    result_df = []
    for e in eps:
        for m in min_samples:
            for a in algorithm_euc:
                db = DBSCAN(metric = distance[0], eps=e, min_samples=m, algorithm=a).fit(d)
                labels = db.labels_
                plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
                plt.title((e,'eps with', m,'min_samples',a,'Algorithm & Euclidean'))
                plt.show()
    cnt += 1


# In[83]:


# Haming
cnt = 0
index = 0
result = []
for d in region_pca_df:
    result_df = []
    for e in eps:
        for m in min_samples:
            for a in algorithm_ha:
                db = DBSCAN(metric = distance[1], eps=e, min_samples=m, algorithm=a).fit(d)
                labels = db.labels_
                plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
                plt.title((e,'eps with', m,'min_samples',a,'Algorithm & Haming'))
                plt.show()
    cnt += 1


# In[ ]:


# Haming
cnt = 0
index = 0
result = []
for d in income_pca_df:
    result_df = []
    for e in eps:
        for m in min_samples:
            for a in algorithm_ha:
                db = DBSCAN(metric = distance[1], eps=e, min_samples=m, algorithm=a).fit(d)
                labels = db.labels_
                plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
                plt.title((e,'eps with', m,'min_samples',a,'Algorithm & Haming'))
                plt.show()
    cnt += 1


# In[ ]:


# Haming
cnt = 0
index = 0
result = []
for d in unit_pca_df:
    result_df = []
    for e in eps:
        for m in min_samples:
            for a in algorithm_ha:
                db = DBSCAN(metric = distance[1], eps=e, min_samples=m, algorithm=a).fit(d)
                labels = db.labels_
                plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
                plt.title((e,'eps with', m,'min_samples',a,'Algorithm & Haming'))
                plt.show()
    cnt += 1


# ## K-means

# In[86]:


# hyper parameter setting
n_cluster = [2,3,4,5,6]
max_iter = [50,100,200,300]


# In[87]:


for d in region_pca_df:
    for n in n_cluster:
        for m in max_iter:
            kmeans = KMeans(n_clusters=n, max_iter=m, random_state=0)
            kmeans.fit(d)
            labels = pd.DataFrame(kmeans.labels_)
            plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
            plt.title((n,'N_cluster with', m,'max_iter'))
            plt.show()


# ## EM

# In[88]:


# hyper parameter setting
n_components = [2,3,4,5,6]
max_iter = [50,100,200,300]


# In[89]:


for d in region_pca_df:
    for n in n_components:
        for m in max_iter:
            gmm = mixture.GaussianMixture(n_components=n, max_iter=m, random_state=0)
            gmm.fit(d)
            labels = gmm.predict(d)
            labels_pd = pd.DataFrame(labels)
            plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
            plt.title((n,'components with', m,'iteratons'))
            plt.show()


# In[90]:


for d in income_pca_df:
    for n in n_components:
        for m in max_iter:
            gmm = mixture.GaussianMixture(n_components=n, max_iter=m, random_state=0)
            gmm.fit(d)
            labels = gmm.predict(d)
            labels_pd = pd.DataFrame(labels)
            plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
            plt.title((n,'components with', m,'iteratons'))
            plt.show()


# In[91]:


for d in unit_pca_df:
    for n in n_components:
        for m in max_iter:
            gmm = mixture.GaussianMixture(n_components=n, max_iter=m, random_state=0)
            gmm.fit(d)
            labels = gmm.predict(d)
            labels_pd = pd.DataFrame(labels)
            plt.scatter(d[d.columns[0]],d[d.columns[1]], c=labels, s=40, cmap='viridis')
            plt.title((n,'components with', m,'iteratons'))
            plt.show()

