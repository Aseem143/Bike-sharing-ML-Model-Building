#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[541]:


# Importing the Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # Dataset Loading

# In[542]:


bs_df = pd.read_csv('day.csv')


# In[543]:


bs_df.head(10)


# # Data Sanity Checks

# In[544]:


bs_df.shape


# In[545]:


bs_df.info()


# # Data Understanding

# In[546]:


bs_df.describe()


# In[547]:


bs_df.columns


# # Missing Value check

# In[548]:


bs_df.isnull().mean()


# In[549]:


# Dropping dteday column as we have already have month and weekday columns
bs_df.drop(['dteday'], axis = 1, inplace = True)


# Droping instant column as it is Id Column which has nothing to do 
bs_df.drop(['instant'], axis = 1, inplace = True)


# Dropping casual and registered columns as we have cnt column as Target variable which is sum of the both 
bs_df.drop(['casual'], axis = 1, inplace = True)
bs_df.drop(['registered'], axis = 1, inplace = True)


# In[550]:


bs_df.head()


# # EDA

# In[551]:


bs_df.nunique()


# In[552]:


bs_df.shape


# In[553]:


cat_cols=['yr','holiday','workingday']
cont_cols=["season","mnth","weekday","weathersit","temp","atemp","hum","windspeed"]
target=["cnt"]
len(cat_cols)+len(cont_cols)+len(target)


# # Univariate Analysis

# In[554]:


bs_df.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)

bs_df.weathersit.replace({1:'good',2:'moderate',3:'bad',4:'severe'},inplace = True)

bs_df.mnth = bs_df.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                  7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'})

bs_df.weekday = bs_df.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'})
bs_df.yr =bs_df.yr.replace({'2018':0,'2019':1})




# In[555]:


bs_df.head(10)


# In[556]:


for i in cont_cols:
    sns.histplot(bs_df[i])
    plt.show()


# In[557]:


bs_df


# In[558]:


sns.heatmap(bs_df[cont_cols].corr(), cmap='flare', annot = True)
plt.show()


# In[559]:


plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
sns.boxplot(x="season",y='cnt',data=bs_df)
plt.subplot(2,4,2)
sns.boxplot(x="weathersit",y='cnt',data=bs_df)
plt.subplot(2,4,3)
sns.boxplot(x="mnth",y="cnt",data=bs_df)
plt.subplot(2,4,4)
sns.boxplot(x="weekday",y="cnt",data=bs_df)
plt.subplot(2,4,5)
sns.boxplot(x="yr",y="cnt",data=bs_df)
plt.subplot(2,4,6)
sns.boxplot(x="workingday",y="cnt",data=bs_df)
plt.subplot(2,4,7)
sns.boxplot(x="yr",y="cnt",data=bs_df)
plt.subplot(2,4,8)
sns.boxplot(x="holiday",y="cnt",data=bs_df)


# In[560]:


for i in cont_cols:
    sns.scatterplot(bs_df[i],bs_df["cnt"])
    plt.show()


# In[561]:


bs_df['season'] = bs_df.season.astype('category')
bs_df['mnth'] = bs_df.mnth.astype('category')
bs_df['weekday'] = bs_df.weekday.astype('category')
bs_df['weathersit'] = bs_df.weathersit.astype('category')


# In[562]:


# Plotting visualisation for season column
x=bs_df['season']
y=bs_df['cnt']
sns.barplot(x='season',y='cnt',hue='yr',data=bs_df)
plt.legend(labels=['2018','2019'])
plt.show()


# In[563]:


plt.figure(figsize=(8,6))
bs_df["mnth"].groupby
sns.barplot(x='mnth',y='cnt',data=bs_df)
plt.show()


# There were most bookings during the months of apr,may,june, july, aug, sept and oct. The trend increased at the start of the year until mid-year, and then it started to decrease as we approached the end of the year. From 2018 to 2019, the number of bookings per month appears to have increased.
# 

# In[564]:


plt.figure(figsize=(8,6))
sns.barplot(x='weathersit',y='cnt',hue='yr',data=bs_df)
plt.legend(labels=['2018','2019'])
plt.show()


# Clear weather attracted more booking which seems obvious. And in comparison to previous year, i.e 2018, booking increased for each weather situation in 2019.

# In[565]:


plt.figure(figsize=(8,6))
sns.barplot(x='weekday',y='cnt',hue='yr',data=bs_df)
plt.legend(labels=['2018','2019'])
plt.show()


# In[566]:


plt.figure(figsize=(8,6))
sns.barplot(x='holiday',y='cnt',hue='yr',data=bs_df)
plt.legend(labels=['2018','2019'])
plt.show()


# When its not holiday, booking seems to be less in number which seems reasonable as on holidays, people may want to spend time at home and enjoy with family.

# In[567]:


# Plotting visualisation for year 
plt.figure(figsize=(8,6))
sns.barplot(x='yr',y='cnt',data=bs_df)
plt.show()


# 2019 attracted more number of booking from the previous year, which shows good progress in terms of business.

# In[568]:


# Analysing/visualizing the numerical columns

sns.pairplot(data=bs_df,vars=['temp','atemp','windspeed','cnt'])
plt.show()


# # Step 3 : Data preparation

# In[569]:


# Dummy variable creation for month, weekday, weathersit and season variables.

months_df=pd.get_dummies(bs_df.mnth,drop_first=True)
weekdays_df=pd.get_dummies(bs_df.weekday,drop_first=True)
weathersit_df=pd.get_dummies(bs_df.weathersit,drop_first=True)
seasons_df=pd.get_dummies(bs_df.season,drop_first=True)


# In[570]:


months_df


# In[571]:


# Merging  the dataframe, with the dummy variable dataset. 
pd.set_option('display.max_columns', 100)
df_new = pd.concat([bs_df,months_df,weekdays_df,weathersit_df,seasons_df],axis=1)


# In[572]:


df_new


# In[573]:


df_new.shape


# In[574]:


df_new.info()


# In[575]:


# dropping unnecessary columns as we have already created dummy variable out of it.

df_new.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)


# In[576]:


df_new.head()


# In[577]:


# checking the shape of new dataframe

df_new.shape


# In[578]:


# checking the column info of new dataframe 

df_new.info()


# # Step 4 : Splitting the data into train and test sets

# In[579]:


# splitting the dataframe into Train and Test

from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(df_new, train_size = 0.7, random_state = 100)


# In[580]:


# check the shape of training datatset

df_train.shape


# In[581]:


df_test.shape


# In[582]:


# Using MinMaxScaler to Rescaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[583]:


# verifying the head of dataset before scaling.
df_train.head()


# In[591]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['temp','atemp','hum','windspeed','cnt']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[592]:


# verifying the head after appying scaling.

df_train.head()


# In[593]:


# describing the dataset

df_train.describe()


# In[594]:


# check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (25, 25))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[595]:


# Visualizing one of the correlation to see the trends via Scatter plot.

plt.figure(figsize=[6,6])
plt.scatter(df_train.temp, df_train.cnt)
plt.show()


# Visualization confirms the positive correlation between temp and cnt.

# In[596]:


# Building the Linear Model

y_train = df_train.pop('cnt')
X_train = df_train


# In[597]:


# Recursive feature elimination and importing necessary libaries for it

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[598]:


lm = LinearRegression()


# In[599]:


df_train


# In[600]:


X_train


# In[601]:


lm.fit(X_train, y_train)


# In[602]:


rfe=RFE(lm,n_features_to_select=15)
rfe=rfe.fit(X_train, y_train)


# In[603]:


#List of variables selected in top 15 list

list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[604]:


# selecting the selected variable via RFE in col list

col = (X_train.columns[rfe.support_])
print(col)


# In[605]:


# Generic function to calculate VIF of variables
from statsmodels.stats.outliers_influence import variance_inflation_factor 
def calculateVIF(df):
    vif = pd.DataFrame()
    vif['Features'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif 


# In[606]:


# dataframe with RFE selected variables

X_train_rfe = X_train[col]


# In[607]:


X_train_rfe


# In[608]:


# calculate VIF

calculateVIF(X_train_rfe)


# # Step 5 : Building a linear model

# In[609]:


# Building 1st linear regression model
import statsmodels.api as sm
X_train_lm_1 = sm.add_constant(X_train_rfe)
lr_1 = sm.OLS(y_train,X_train_lm_1).fit()
print(lr_1.summary())


# In[610]:


# As workingday shows high VIF values hence we can drop it
X_train_new = X_train_rfe.drop(['workingday'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[611]:


# Building 2nd linear regression model

X_train_lm_2 = sm.add_constant(X_train_new)
lr_2 = sm.OLS(y_train,X_train_lm_2).fit()
print(lr_2.summary())


# As humidity shows high VIF values hence we can drop it

# In[612]:


X_train_new = X_train_rfe.drop(['hum'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[613]:


# Building 3rd linear regression model

X_train_lm_3 = sm.add_constant(X_train_new)
lr_3 = sm.OLS(y_train,X_train_lm_3).fit()
print(lr_3.summary())


# In[614]:


# We can drop dec variable as it has high p-value
X_train_new = X_train_new.drop(['sat'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[615]:


# Building 4th linear regression model

X_train_lm_4 = sm.add_constant(X_train_new)
lr_4 = sm.OLS(y_train,X_train_lm_4).fit()
print(lr_4.summary())


# In[616]:


# We can drop dec variable as it has high p-value
X_train_new = X_train_new.drop(['good'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[617]:


X_train_lm_4 = sm.add_constant(X_train_new)
lr_4 = sm.OLS(y_train,X_train_lm_4).fit()
print(lr_4.summary())


# In[618]:


X_train_new = X_train_new.drop(['jul'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[619]:


# Building 5th linear regression model

X_train_lm_5 = sm.add_constant(X_train_new)
lr_5 = sm.OLS(y_train,X_train_lm_5).fit()
print(lr_5.summary())


# In[620]:


X_train_new = X_train_new.drop(['workingday'], axis = 1)

calculateVIF(X_train_new)


# In[621]:


# Building 6th linear regression model

X_train_lm_6 = sm.add_constant(X_train_new)
lr_6 = sm.OLS(y_train,X_train_lm_6).fit()
print(lr_6.summary())


# In[622]:


X_train_new = X_train_new.drop(['windspeed'], axis = 1)

calculateVIF(X_train_new)


# In[623]:


# Building 7th linear regression model

X_train_lm_7 = sm.add_constant(X_train_new)
lr_7 = sm.OLS(y_train,X_train_lm_7).fit()
print(lr_7.summary())


# In[624]:


X_train_new = X_train_new.drop(['spring'], axis = 1)
calculateVIF(X_train_new)


# We can cosider the above model i.e lr_7, as it seems to have very low multicolinearity between the predictors and the p-values for all the predictors seems to be significant.
# 

# # Step 6: Residual Analysis of the train data and validation
# 

# In[625]:


X_train_lm_7


# In[626]:


y_train_pred = lr_7.predict(X_train_lm_7)


# # Normality of error terms

# In[627]:


# Plot the histogram of the error terms

fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20) 
plt.xlabel('Errors', fontsize = 18)


# # Multi Colinearity

# In[628]:


calculateVIF(X_train_new)


# In[629]:


plt.figure(figsize=(15,8))
sns.heatmap(X_train_new.corr(),annot = True, cmap="RdYlGn")
plt.show()


# VIF values are less than 5 which is good and also there is no multicolinearity as seen from the heatmap.

# In[630]:


# Linear relationship validation using CCPR plot
# Component and component plus residual plot

sm.graphics.plot_ccpr(lr_7, 'yr')
plt.show()

sm.graphics.plot_ccpr(lr_7, 'sept')
plt.show()

sm.graphics.plot_ccpr(lr_7, 'winter')
plt.show()


# # Homoscedasticity

# In[631]:


sns.regplot(x=y_train, y=y_train_pred)
plt.title('Predicted Points Vs. Actual Points', fontdict={'fontsize': 20})
plt.xlabel('Actual Points', fontdict={'fontsize': 15})
plt.ylabel('Predicted Points', fontdict={'fontsize': 15})
plt.show()


# From the above graph, we can say that residuals are equal distributed across predicted value.
# This means we see equal variance and we do not observe high concentration of data points in certain region & low concentarion in certain regions.
# This prooves Homoscedasticity of Error Terms

# # Step 7: Making Predictions Using the Final Model
# 
# 

# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final, i.e. 7th model.
# 

# In[633]:


# Apply scaler() to all numeric variables in test dataset. Note: we will only use scaler.transform, 
# as we want to use the metrics that the model learned from the training data to be applied on the test data. 
# In other words, we want to prevent the information leak from train to test dataset.

num_vars = ['temp','atemp','hum','windspeed','cnt']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()


# # Dividing into X_test and y_test

# In[634]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[635]:


#Selecting the variables that were part of final model.
col1 = X_train_new.columns

X_test = X_test[col1]



# In[650]:


# Adding constant variable to test dataframe
X_test_lm_7 = sm.add_constant(df_test)


# In[653]:


X_test_lm_7


# In[ ]:




