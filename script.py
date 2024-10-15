#!/usr/bin/env python
# coding: utf-8

# In[16]:


import bamboolib as bam 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import statsmodels.api as sm  
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import time
import math
from random import uniform
from scipy.stats import  randint as sp_randint
import urllib.request
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')


# In[17]:


df=pd.read_excel('17_Batch_MasterData.xlsx', skiprows=2)
df.head()


# In[18]:


# Checking the Size of the dataset 
df.shape


# ##### The dataset has 209 records and 30 attributes

# In[19]:


# Checking the Data types
df.dtypes


# ##### Most of the attributes are object type which needs to be changed to appropriate string or float.

# In[20]:


df.info() 


# In[21]:


# Identifying the columns 
df.columns


# In[22]:


# Summary of dataset
df.describe()


# ### Checking missing values 

# In[23]:


# percentage of missing values per column
(df.isna().sum() / len(df)) * 100


# ##### Lot of missing values in the dataset that needs to be corrected. 

# In[24]:


# Check to see if 'id' is unique identifier for each sample
print('Sum of duplicate values:{}\n'.format(df[' ROLL NO'].duplicated().sum()))


# ##### No duplicates in the dataset

# ### Data Cleaning

# ##### Dropping columns 'S.No', 'STUDENT NAME WITHOUT SURNAME', 'FIRST LETTER OF SURNAME' as they are redundent
# 

# In[25]:


df = df.drop(columns=['S.No', 'STUDENT NAME WITHOUT SURNAME', 'FIRST LETTER OF SURNAME'])


# ##### Converting columns to string where required and renaming " Roll No" to "Roll.No"

# In[26]:


df = df.rename(columns={' ROLL NO': 'ROLL/NO'})
df['ROLL/NO'] = df['ROLL/NO'].astype('string')
df['STUDENT NAME'] = df['STUDENT NAME'].astype('string')
df['GENDER'] = df['GENDER'].astype('string')
df['BRANCH'] = df['BRANCH'].astype('string')
df['10TH BOARD NAME'] = df['10TH BOARD NAME'].astype('string')
df['10TH SCHOOL NAME'] = df['10TH SCHOOL NAME'].astype('string')
df['12TH / DIPLOMA BOARD NAME'] = df['12TH / DIPLOMA BOARD NAME'].astype('string')
df['Entrance Mode for Admission into VNR'] = df['Entrance Mode for Admission into VNR'].astype('string')
df['STUDENT CITY'] = df['STUDENT CITY'].astype('string')
df['Registration for Placements/ Higher Education'] = df['Registration for Placements/ Higher Education'].astype('string')
df['CATEGORY'] = df['CATEGORY'].astype('string')


# In[27]:


df['10TH CGPA'] = pd.to_numeric(df['10TH CGPA / %'], downcast='float', errors='coerce')
df.loc[df['10TH CGPA'] > 10, '10TH CGPA'] = df['10TH CGPA']/10
df[['10TH CGPA']] = df[['10TH CGPA']].fillna(df[['10TH CGPA']].median())

df['INTER / DIPLOMA CGPA'] = pd.to_numeric(df['INTER / DIPLOMA %'], downcast='float', errors='coerce')
df.loc[df['INTER / DIPLOMA CGPA'] > 10, 'INTER / DIPLOMA CGPA'] = df['INTER / DIPLOMA CGPA']/10
df[['INTER / DIPLOMA CGPA']] = df[['INTER / DIPLOMA CGPA']].fillna(df[['INTER / DIPLOMA CGPA']].median())

df['BTECH CGPA'] = pd.to_numeric(df['B.TECH CGPA UPTO\n 2-2'], downcast='float', errors='coerce')
df['BTECH CGPA'] = pd.to_numeric(df['BTECH CGPA'], downcast='float', errors='coerce')
df[['BTECH CGPA']] = df[['BTECH CGPA']].fillna(df[['BTECH CGPA']].median())

df[['Number of ACTIVE BACKLOGS', 'NO OF PASSIVE BACKLOGS']] = df[['Number of ACTIVE BACKLOGS', 'NO OF PASSIVE BACKLOGS']].fillna(0)

df[['12TH / DIPLOMA BOARD NAME']] = df[['12TH / DIPLOMA BOARD NAME']].fillna('Not Known')

df[['Entrance Mode for Admission into VNR']] = df[['Entrance Mode for Admission into VNR']].fillna(df[['Entrance Mode for Admission into VNR']].mode().iloc[0])

df[['EAMCET RANK', 'JEE RANK', 'ECET RANK']] = df[['EAMCET RANK', 'JEE RANK', 'ECET RANK']].fillna(0)
df['EAMCET RANK'] = pd.to_numeric(df['EAMCET RANK'], downcast='integer', errors='coerce')
df['ECET RANK'] = pd.to_numeric(df['ECET RANK'], downcast='integer', errors='coerce')
df['JEE RANK'] = pd.to_numeric(df['JEE RANK'], downcast='integer', errors='coerce')

df[['Registration for Placements/ Higher Education']] = df[['Registration for Placements/ Higher Education']].fillna(df[['Registration for Placements/ Higher Education']].mode().iloc[0])

df["B.TECH I-I SEM SGPA"] = df["B.TECH I-I SEM SGPA"].str.replace('.8.96', '8.96', regex=False)
df[['B.TECH I-I SEM SGPA']] = df[['B.TECH I-I SEM SGPA']].fillna(0)
df['B.TECH I-I SEM SGPA'] = pd.to_numeric(df['B.TECH I-I SEM SGPA'], downcast='float', errors='coerce')
df[['B.TECH I-II SEM SGPA']] = df[['B.TECH I-II SEM SGPA']].fillna(0)
df[['B.TECH 2-1 SGPA', 'B.TECH 2-2 SGPA', 'B.TECH CGPA TILL DATE']] = df[['B.TECH 2-1 SGPA', 'B.TECH 2-2 SGPA', 'B.TECH CGPA TILL DATE']].fillna(0)
df['B.TECH I-II SEM SGPA'] = pd.to_numeric(df['B.TECH I-II SEM SGPA'], downcast='float', errors='coerce')
df['B.TECH 2-1 SGPA'] = pd.to_numeric(df['B.TECH 2-1 SGPA'], downcast='float', errors='coerce')
df['B.TECH 2-1 SGPA'] = pd.to_numeric(df['B.TECH 2-1 SGPA'], downcast='float', errors='coerce')
df['B.TECH 2-2 SGPA'] = pd.to_numeric(df['B.TECH 2-2 SGPA'], downcast='float', errors='coerce')
df['B.TECH CGPA TILL DATE'] = pd.to_numeric(df['B.TECH CGPA TILL DATE'], downcast='float', errors='coerce')
df.head()


# In[28]:


df['Total Backlogs'] = df['Number of ACTIVE BACKLOGS'] + df['NO OF PASSIVE BACKLOGS']


# In[34]:


df.describe()


# ### Exploratory Data Analysis using Data Visualizations 

# #### Histograms to see spraed of data

# In[29]:


# Histogram of numerical values
df.hist(figsize=(13,11),bins=20)
plt.show()


# In[30]:


fig, axs = plt.subplots(2, 2, figsize=(16, 16))

sns.histplot(x= "BTECH CGPA",
             data=df,
              bins=25,
              hue="GENDER",
              kde=True, ax=axs[0, 0])

sns.histplot(x= "Total Backlogs",
             data=df,
              bins=25,
              hue="GENDER",
              kde=True , ax=axs[0, 1])

sns.histplot(x= "BTECH CGPA",
             data=df,
              bins=50,
              hue="CATEGORY",
              kde=True , ax=axs[1, 0])

sns.histplot(x= "Total Backlogs",
             data=df,
              bins=25,
              hue="CATEGORY",
              kde=True , ax=axs[1, 1])


# In[31]:


plt.figure(figsize = (18, 8))
sns.distplot(df['Number of ACTIVE BACKLOGS'])

# Majority of the students have less than 2 active backlogs


# In[32]:


plt.figure(figsize = (18, 8))
sns.distplot(df['NO OF PASSIVE BACKLOGS'])

# Majority of the students have less than 1 passive backlogs


# ##### Count plots for categorical variables 

# In[33]:


# dataset exploration based on categorical values
fig, ax = plt.subplots(2,2, figsize=(16,16))
a=sns.countplot(data=df, x='GENDER' , ax=ax[0,0])
for container in a.containers:
    a.bar_label(container)
a=sns.countplot(data=df, x='Entrance Mode for Admission into VNR' , ax=ax[0,1])
for container in a.containers:
    a.bar_label(container)
a=sns.countplot(data=df, x='CATEGORY' , ax=ax[1,0])
for container in a.containers:
    a.bar_label(container)
a=sns.countplot(data=df, x='Registration for Placements/ Higher Education' , ax=ax[1,1])
for container in a.containers:
    a.bar_label(container)


# In[35]:


# 10th board that accounts to highest number of backlogs
fig = px.bar(df, x='10TH BOARD NAME', y='Total Backlogs',  color='GENDER', barmode='group',)
fig.show()
# SSC board shows the highest number of backlogs 


# In[36]:


# 12th board that accounts to highest number of backlogs
fig = px.bar(df, x='12TH / DIPLOMA BOARD NAME', y='Total Backlogs',  color='GENDER', barmode='group',)
fig.show()
# SBTET and BOI boards show the highest number of backlogs 


# In[37]:


# CGPA trend analysis of VNR students for 10th Inter/Diploma and Btech

wtl_col = ['STUDENT NAME','10TH CGPA','INTER / DIPLOMA CGPA','BTECH CGPA']
wtl = df[wtl_col]
wtl = pd.melt(wtl, id_vars=['STUDENT NAME'], value_vars=['STUDENT NAME','10TH CGPA','INTER / DIPLOMA CGPA','BTECH CGPA'])
wtl


# In[38]:


avg1 = np.average(wtl[wtl['variable']=='10TH CGPA']['value'])
avg2 = np.average(wtl[wtl['variable']=='INTER / DIPLOMA CGPA']['value'])
avg3 = np.average(wtl[wtl['variable']=='BTECH CGPA']['value'])

fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(data=wtl, y='value',x='variable', hue='variable')
plt.plot(['10TH CGPA','INTER / DIPLOMA CGPA', 'BTECH CGPA'],[avg1,avg2,avg3])

plt.title("The average CGPA is geting lower", fontsize = 18)
plt.suptitle('Changes for average CGPA', fontsize = 22)
plt.show()


# In[42]:


# 10th cgpa versus total backlogs 

sns.lmplot(data=df,x='10TH CGPA',y='Total Backlogs', hue='GENDER',height=8, aspect=1.5)
plt.title("Relationship between 10th grade CGPA and total backlogs", fontsize = 18)
plt.suptitle('The lower the gpa in 10th more are the BTECH backlogs', fontsize = 14)
plt.show()



# In[43]:


# 12th cgpa versus total backlogs 

sns.lmplot(data=df,x='INTER / DIPLOMA CGPA',y='Total Backlogs', hue='GENDER',height=8, aspect=1.5 )
plt.title("Relationship between INTER CGPA and total backlogs", fontsize = 18)
plt.suptitle('The lower the gpa in Inter/Diploma more are the BTECH backlogs', fontsize = 14)
plt.show()


# In[41]:


sns.lmplot(data=df,x='INTER / DIPLOMA CGPA',y='BTECH CGPA', hue='GENDER',height=8, aspect=1.5)
plt.title("Relationship between INTER CGPA and Bachelor CGPA", fontsize = 18)
plt.suptitle('They are positively related', fontsize = 14)

plt.show()


# ### Outliers

# In[44]:


df.columns


# In[45]:


df_describe = df.drop(columns = ['ROLL/NO', 'STUDENT NAME', 'GENDER', 'BRANCH',
                                 '10TH BOARD NAME', '10TH YEAR OF PASSING', '10TH SCHOOL NAME',
                                 '12TH / DIPLOMA BOARD NAME', '12TH / DIPLOMA YEAR OF PASSING',
                                 '12TH / DIPLOMA COLLEGE NAME', 'Entrance Mode for Admission into VNR',
                                 'CATEGORY', 'STUDENT CITY',
                                 'Registration for Placements/ Higher Education',
                                 '10TH CGPA / %','INTER / DIPLOMA %', 'B.TECH CGPA UPTO\n 2-2'])

plt.figure(figsize =(20, 20 ))
x = 1 

for column in df_describe.columns:
    plt.subplot(4,5,x)
    sns.boxplot(df_describe[column])
    x+=1
plt.tight_layout
plt.show()


# In[46]:


fig = go.Figure()
fig.add_trace(go.Box(y=df["10TH CGPA"], name="10TH CGPA"))
fig.add_trace(go.Box(y=df["INTER / DIPLOMA CGPA"], name="INTER / DIPLOMA CGPA"))
fig.add_trace(go.Box(y=df["BTECH CGPA"], name="BTECH CGPA"))

fig.show()


# ### Correlation Analysis

# In[47]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')





