'''
Project Title : Bank Marketing
Author : Lee Ye Jin
Last Modified : 2020.11.
'''

import os
import random
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Dataset
data = pd.read_csv('../Data/bank-full.csv')
print(data.info())
print(data.shape)

# Make dirty data
for i in range(len(data)):
    p = random.randint(1, 100)
    if p < 6:
        k = random.randint(0, len(data.columns.tolist())-1)
        data.iloc[i, k] = None

data.to_csv(os.getcwd() + "\\bank_dirty.csv", mode='w', index=False)  # save it as dirty version

# Data Exploration
# Count of null values
bank = pd.read_csv('bank_dirty.csv')
missing = bank.isnull()
print(missing.sum())

# Clear dirty data
bank = bank.dropna(axis=0)
print(bank.isnull().sum())

# Explore numerical values
fig, axs = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(20, 15))

counter = 0
num_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for num_column in num_columns:
    trace_x = counter // 4
    trace_y = counter % 4

    axs[trace_x, trace_y].boxplot(bank[num_column])
    axs[trace_x, trace_y].set_title(num_column)

    counter += 1

plt.show()

# Remove outliers
print(bank.groupby(['deposit', 'pdays']).size())
# Because pdays has -1 value too much
bank.drop('pdays', axis=1, inplace=True)

# Age value is variable -> This column can be ignored
print(bank.groupby('age', sort=True)['age'].count())
bank.drop('age', axis=1, inplace=True)

print(bank.groupby(['deposit', 'balance'], sort=True)['balance'].count())
print(bank.groupby(['deposit', 'duration'], sort=True)['duration'].count())

print(bank.groupby(['deposit', 'campaign'], sort=True)['campaign'].count())
bank = bank[bank['campaign'] < 33]

print(bank.groupby(['deposit', 'previous'], sort=True)['previous'].count())
bank = bank[bank['previous'] < 31]

# Explore categorical values
print(bank.shape)
cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(18, 15))

counter = 0
for cat_column in cat_columns:
    value_counts = bank[cat_column].value_counts()

    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))

    axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label=value_counts.index)

    axs[trace_x, trace_y].set_title(cat_column)

    for tick in axs[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)

    counter += 1

plt.show()

# Because default has 'No' value too much
bank.drop('default', axis=1, inplace=True)

print(bank.poutcome.value_counts())
# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank['poutcome'] = bank['poutcome'].replace(['other'], 'unknown')
print(bank.poutcome.value_counts())

# every customer has been contacted -> No effect to target
bank.drop('contact', axis=1, inplace=True)

# Day and Month value -> No effect to target
# Categorical data
bank.drop('month', axis=1, inplace=True)
# Numerical data
bank.drop('day', axis=1, inplace=True)
bank.drop('job', axis=1, inplace=True)

bank.columns = ['marital', 'education', 'balance', 'housing', 'loan', 'duration', 'campaign', 'previous', 'poutcome', 'deposit']
bank.to_csv('./Data/After_bank.csv', header=False, index=False)
