'''
Project Title : Bank Marketing
Author : Lee Ye Jin
Last Modified : 2020.11.21
'''

import random
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Import Dataset
data = pd.read_csv('C:/Users/User/Desktop/bank/bank.csv')
print('Check the dataset>> ', data.info())
print('Check the dataset shape>> ', data.shape)

# Make dirty data
for i in range(len(data)):
    p = random.randint(1, 100)
    if p < 6:
        k = random.randint(0, len(data.columns.tolist())-1)
        data.iloc[i, k] = None

data.to_csv('C:/Users/User/Desktop/bank/DirtyBank.csv', index=False)

# Data Exploration
# Count of null values
bank = pd.read_csv('C:/Users/User/Desktop/bank/DirtyBank.csv')
missing = bank.isnull()
print('\nCheck the sum of missing value>>\n', missing.sum())

# Clear dirty data
bank = bank.dropna(axis=0)
print('\nAfter drop missing value>>\n', bank.isnull().sum())

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
# Age column can be ignored
print(bank.groupby(['deposit', 'age'], sort=True)['age'].size())
bank.drop('age', axis=1, inplace=True)

print(bank.groupby(['deposit', 'pdays']).size())
# pdays -1 values are too much
bank.drop('pdays', axis=1, inplace=True)

# Balance outlier should not be removed(Balance goes high -> client has more interest)
print(bank.groupby(['deposit', 'balance'], sort=True)['balance'].size())

# Duration same as Balance
print(bank.groupby(['deposit', 'duration'], sort=True)['duration'].size())

# Assume >= 33 are outlier
print(bank.groupby(['deposit', 'campaign'], sort=True)['campaign'].size())
bank = bank[bank['campaign'] < 33]

# Assume >= 31 are outlier
print(bank.groupby(['deposit', 'previous'], sort=True)['previous'].size())
bank = bank[bank['previous'] < 31]

# After removing outliers in Numeric values
print('\nCheck the dataset shape after removing>> ', bank.shape)

# Explore categorical values
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

# Relationship between categorical values and target value(Deposit)
for cat_column in cat_columns:
    sns.catplot(x='deposit', col=cat_column, kind='count', data=bank)
plt.show()

# Remove outliers
# job values have Unknown(It is not important info and should be removed)
bank = bank[bank.job != 'unknown']

# marital values have Unknown(It is not important info and should be removed)
bank = bank[bank.marital != 'unknown']

# education values have very small 'illiterate'(only 18)
bank = bank[bank.marital != 'illiterate']

# Save the dataset name: After_bank.csv
bank.to_csv('C:/Users/User/Desktop/bank/After_bank.csv', index=False)
