# coding: utf-8

# packages
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np


# reading in the wiki data from csv file
dataWiki = pd.read_csv('output.csv')

# verifying the upload was successful
print(dataWiki.shape)

# reading in the Kaggle data from csv file
dataKaggle = pd.read_csv('Data_Breaches_EN_V2_2004_2017_20180220.csv', sep=';')

# verifying the upload was successful
print(dataKaggle.shape)

# summary stats for wiki data
dataWiki.describe(include='all')

# summary stats for Kaggle data
dataKaggle.describe(include='all')

# list of all cols in dataset
print(list(dataKaggle.columns))

# checking for null values in wiki data
null_columns=dataWiki.columns[dataWiki.isnull().any()]
dataWiki[null_columns].isnull().sum()

# checking for null values in Kaggle data
null_columns=dataKaggle.columns[dataKaggle.isnull().any()]
dataKaggle[null_columns].isnull().sum()

# replacing Nan values with unknown
dataWiki['Records'] = dataWiki['Records'].fillna('unknown')

# replacing Nan values with none
dataWiki['Organization type'] = dataWiki['Organization type'].fillna('none')

# replacing Nan values with none
dataWiki['Method'] = dataWiki['Method'].fillna('none')

# acquiring list of unique values in the Records col
print(dataWiki['Records'].unique())

# values that need to be removed, since it's only a handful we will be removing the whole row
filteredDataValues = ['none', 'millions', '235 GB', '250 locations', '500 locations',
                     '10 locations', '93 stores', 'undisclosed', '2.3 million',
                     '100 terabytes', '54 locations', '200 stores', '8 locations',
                     '51 locations', '19 years of data']

# creating filter to drop rows from DF
dropRows = dataWiki[dataWiki['Records'].isin(filteredDataValues)]

# drop rows from wiki DF if they are in the dropRows filter
dataWiki = dataWiki[~dataWiki.index.isin(dropRows.index)]

# sanity check to ensure I didn't delete everything
print(dataWiki.shape)

# creating a pivot table to view data by org and method of data breach
dataWiki.groupby(['Organization type', 'Method']).sum().sort_values('Records',ascending=False)

# cleansing Records Lost col to run visualizations on, replacing all commas with a period
dataKaggle['Records Lost'] = dataKaggle['Records Lost'].apply(lambda x: str(x.replace(',','.')))

# converting string to float data type for col
#dataKaggle['Records Lost'] = dataKaggle['Records Lost'].apply(lambda x: format(float(x)))

# converting string to float data type for col
dataKaggle['Records Lost'] = pd.to_numeric(dataKaggle['Records Lost'])

# creating a pivot table to view Kaggle data by sector == Org(Wiki) and method of leak == Method(wiki)
dataKaggle.groupby(['Sector','Method of Leak']).sum().sort_values('Records Lost', ascending=False)

# creating my encoder to convert cat variable to numeric type
lb_make = LabelEncoder()

# converting cat data
dataWiki['Organization type'] = lb_make.fit_transform(dataWiki["Organization type"])

#converting cat data
dataWiki['Method'] = lb_make.fit_transform(dataWiki["Method"])

# creating a copy of main dataframe to work on without destroying main data set
dataWikiSub = dataWiki[['Organization type', 'Method', 'Records']].copy()

# converting cat data
dataWikiSub['Records'] = lb_make.fit_transform(dataWiki["Records"])

# plotting correlation matrix between all variables in subDF
f, ax = plt.subplots(figsize=(11, 9))
ax = sns.heatmap(dataWikiSub, cmap="YlGnBu")

corr = dataWikiSub.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

kaggleSubDF = dataKaggle[['Alternative Name', 'Year', 'Records Lost', 'Sector', 'Method of Leak']]

kaggleSubDF["Sector"] = lb_make.fit_transform(kaggleSubDF["Sector"])

kaggleSubDF["Method of Leak"] = lb_make.fit_transform(kaggleSubDF["Method of Leak"])

corr = kaggleSubDF.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

