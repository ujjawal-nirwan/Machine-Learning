#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Basic info of the dataset

# In[2]:


# Loading the dataset
data = pd.read_csv("Marketing_train.csv")
data.info()


# In[3]:


data.describe()


# In[4]:


data.head()


# In[5]:


print(data["profession"].value_counts())
print("*"*25)
print(data["marital"].value_counts())
print("*"*25)
print(data["schooling"].value_counts())


# In[6]:


print(data["responded"].value_counts())


# From the above distribution we can be sure that the data is imbalanced, as the number of "no"s are also 8 times the number of "yes"

# # Exploratory Data Analysis

# ### Distribution of Class variable

# In[7]:


plt.figure(figsize=(8,6))
Y = data["responded"]
total = len(Y)*1.
ax=sns.countplot(x="responded", data=data)
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))
    
ax.yaxis.set_ticks(np.linspace(0, total, 11))
  #adjust the ticklabel to the desired format, without changing the position of the ticks.
ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  # ax.legend(labels=["no","yes"])
plt.show()


# # Univariate Analysis

# In[8]:


def countplot(label, dataset):
    plt.figure(figsize=(15,10))
    Y = data[label]
    total = len(Y)*1.
    ax=sns.countplot(x=label, data=dataset)
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

  #put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
    ax.yaxis.set_ticks(np.linspace(0, total, 11))
  #adjust the ticklabel to the desired format, without changing the position of the ticks.
    ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  # ax.legend(labels=["no","yes"])
    plt.show()


# In[9]:



get_ipython().run_line_magic('matplotlib', 'inline')

def countplot_withY(label, dataset):
    plt.figure(figsize=(20,10))
    Y = data[label]
    total = len(Y)*1.
    ax=sns.countplot(x=label, data=dataset, hue="responded")
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

  #put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
    ax.yaxis.set_ticks(np.linspace(0, total, 11))
  #adjust the ticklabel to the desired format, without changing the position of the ticks.
    ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  # ax.legend(labels=["no","yes"])
    plt.show()


# ## Feature: Job (Categorical variable)

# In[10]:


countplot("profession", data)


# From the above distribution we can see that most of the customers have jobs as "admin", "blue-collar" or "technician". One interesting thing to find out would be to see the distribution for each classes as well. For example, how many people who work as an admin have subscribed a term deposit.

# In[11]:


countplot_withY("profession", data)


# From the above plot, we can see that the customers who have a job of admin have the highest rate of subscribing a term deposit, but they are also the highest when it comes to not subscribing. This is simply because we have more customers working as admin than any other profession. 
# 
# We can find out the odds or ratio of subscribing and not subscribing based on the profession, to find out which profession has the highest odds of subscribing given the data. At this point we are not sure if there is any correlation between job and target variable.
# 
# **Idea:** If we find that odds of one profession subscribing is greater than other job, we can use the odds or log(odds) as a feature by replacing jobs field with the odds, instead of doing one hot encoding.

# ## Feature: Marital (Categorical feature)

# In[12]:


countplot("marital", data)


# In[13]:


countplot_withY("marital", data)


# ## Feature: default (categorical)
# 
# This is a categorical feature which means "has credit in default", with the values "yes" and "no" and "unknown".

# In[14]:


countplot("default", data)


# In[15]:


countplot_withY("default", data)


# There is no customer with who has credit in default. Majority of the customers don't have, and the for the rest of the customers this field is unknown.

# ## Feature: Education

# In[16]:


countplot("schooling",data)


# In[17]:


countplot_withY("schooling", data)


# ## Feature: housing (Categorical)

# In[18]:


countplot("housing", data)


# Majority of the customers have a housing loan.

# In[19]:


countplot_withY("housing", data)


# ## Feature: loan (Categorical)

# In[20]:


countplot("loan", data)


# In[21]:


countplot_withY("loan", data)


# ## Feature: contact (Categorical)

# In[22]:


countplot("contact", data)


# In[23]:


countplot_withY("contact", data)


# ## Feature: month (Categorical)

# In[24]:


countplot("month", data)


# In[25]:


countplot_withY("month", data)


# ## Feature: day_of_week (Categorical)

# In[26]:


countplot("day_of_week", data)


# In[27]:


countplot_withY("day_of_week", data)


# The day of the week seems to be irrelevent as we have the same amount of data for all the days of the week, and no:yes ratio is also almost same.

# ## Feature: poutcome (Categorical)
# 
# This feature indicates the outcome of the previous marketing campaign

# In[28]:


countplot("poutcome", data)


# In[29]:


countplot_withY("poutcome", data)


# ## Feature: Age (Numeric)

# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="custAge")
plt.show()


# From the above boxplot we know that for both the customers that subscibed or didn't subscribe a term deposit, has a median age of around 38-40. And the boxplot for both the classes overlap quite a lot, which means that age isn't necessarily a good indicator for which customer will subscribe and which customer will not.

# In[31]:


plt.figure(figsize=(10,8))
sns.distplot(data["custAge"])


# As we can see in the above distribution also, that most of the customers are in the age range of 20-60.

# This seems like a powerlaw distribution where most the values are very low and very few have high values.

# ## Feature: campaign (numeric)

# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="campaign")
plt.show()


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data["campaign"])
plt.show()


# ## Feature: pdays (numeric)

# In[34]:


data["pdays"].unique()


# In[35]:


data["pdays"].value_counts()


# Most of the values are 999, which means that the most of the customers have never been contacted before.

# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="pdays")
plt.show()


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data[data["responded"]=="yes"]["pdays"])
sns.distplot(data[data["responded"]=="no"]["pdays"])
plt.show()


# ## Feature: previous (numeric)

# In[38]:


data["previous"].unique()


# In[39]:


data["previous"].value_counts()


# In[40]:


data[data["responded"]=="yes"]["previous"].value_counts()


# In[41]:


data[data["responded"]=="no"]["previous"].value_counts()


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="previous")
plt.show()


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data["previous"])
plt.show()


# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data[data["responded"]=="yes"]["previous"])
sns.distplot(data[data["responded"]=="no"]["previous"])
plt.show()


# The previous feature is very similarly distributed for both the classes in the target variable. From basic EDA it is not sure how much value this individual feature have on the target variable.

# In[45]:


countplot("previous", data)


# In[46]:


countplot_withY("previous", data)


# ## emp.var.rate

# In[47]:


data["emp.var.rate"].value_counts()


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="emp.var.rate")
plt.show()


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data["emp.var.rate"])
plt.show()


# ## cons.price.idx

# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="cons.price.idx")
plt.show()


# In[51]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data["cons.price.idx"])
plt.show()


# ## cons.conf.idx

# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="cons.conf.idx")
plt.show()


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data["cons.conf.idx"])
plt.show()


# ## euribor3m

# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="euribor3m")
plt.show()


# In[55]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data["euribor3m"])
plt.show()


# ## nr.employed

# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="responded", y="nr.employed")
plt.show()


# In[57]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
sns.distplot(data["nr.employed"])
plt.show()


# ## Correlation matrix of numerical features

# In[58]:


# Idea of correlation matrix of numerical feature: https://medium.com/datadriveninvestor/introduction-to-exploratory-data-analysis-682eb64063ff
get_ipython().run_line_magic('matplotlib', 'inline')
corr = data.corr()

f, ax = plt.subplots(figsize=(10,12))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

_ = sns.heatmap(corr, cmap="BuPu", square=True, ax=ax, annot=True, linewidth=0.1)

plt.title("Pearson correlation of Features", y=1.05, size=15)


# From the above heatmap we can see that there are some numerical features which share a high correlation between them, e.g nr.employed and euribor3m these features share a correlation value of 0.95, and euribor3m and emp.var.rate share a correlation of 0.97, which is very high compared to the other features that we see in the heatmap.

# # Data Preprocessing

# ## Dealing with duplicate data

# In[59]:


data_dup = data[data.duplicated(keep="last")]
data_dup


# In[60]:


data_dup.shape


# In[61]:


data = data.drop_duplicates()
data.shape


# ## Missing Value Imputation

# In[62]:


data.isnull().sum()


# As a numerical variable we can use mean or median to impute the missing values. We will use median to fill the null values as age columns has outliers

# In[63]:


data['custAge'].fillna(data['custAge'].median(),inplace=True)
data['schooling'].fillna(data['schooling'].mode()[0], inplace=True)
data['day_of_week'].fillna(data['day_of_week'].mode()[0], inplace=True)


# In[64]:


data.isnull().sum()


# ## Logistic Regression

# In[65]:


x_features=list(data.columns)


# In[66]:


x_features.remove('responded')


# In[67]:


data['responded'] = data['responded'].map(lambda x : 1 if x == 'yes' else 0)


# In[68]:


encoded_data = pd.get_dummies(data[x_features], drop_first = True)


# In[69]:


encoded_data.head()


# In[70]:


y=data['responded']
x=encoded_data


# In[71]:


x=pd.get_dummies(x)
data=pd.get_dummies(data)


# In[72]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size =0.8, random_state=42)


# In[73]:


from sklearn.linear_model import LogisticRegression
log_regn = LogisticRegression(max_iter = 200)
log_regn.fit(x_train, y_train)


# In[75]:


y_pred=log_regn.predict(x_test)


# In[79]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# In[80]:


print(metrics.classification_report(y_test, y_pred))


# In[82]:


from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
x_train_u,y_train_u = resample(x_train[y_train == 1],
                              y_train[y_train ==1],
                              n_samples = x_train[y_train==0].shape[0],
                              random_state = 1)


# In[83]:


x_train[y_train == 1]


# In[84]:


y_train[y_train ==1]


# In[85]:


x_train_u = np.concatenate((x_train[y_train == 0], x_train_u))


# In[86]:


x_train.shape


# In[87]:


x_train_u


# In[88]:


y_train_u = np.concatenate((y_train[y_train == 0],y_train_u))


# In[89]:


print(x_train_u.shape)
print(y_train_u.shape)


# In[90]:


log_reg_up = LogisticRegression()


# In[91]:


log_reg_up.fit(x_train_u,y_train_u)


# In[92]:


print(metrics.classification_report(y_test,log_reg_up.predict(x_test)))


# In[93]:


sm = SMOTE(random_state = 12)
x_train_sm,y_train_sm = sm.fit_resample(x_train,y_train)


# In[94]:


log_reg_sm = LogisticRegression(max_iter = 500)


# In[95]:


log_reg_sm.fit(x_train_sm,y_train_sm)


# In[96]:


print(metrics.classification_report(y_test,log_reg_sm.predict(x_test)))


# In[97]:


x_train_d, y_train_d = resample(x_train[y_train == 0],
                               y_train[y_train ==0],
                               n_samples = x_train[y_train ==1].shape[0],
                                random_state=1)


# In[98]:


x_train_d = np.concatenate((x_train[y_train == 1],x_train_d))
y_train_d = np.concatenate((y_train[y_train == 1],y_train_d))


# In[99]:


log_reg_d = LogisticRegression(max_iter = 500)


# In[100]:


log_reg_d.fit(x_train_d,y_train_d)


# In[101]:


print(metrics.classification_report(y_test,log_reg_d.predict(x_test)))


# In[105]:


def confusion_matrix(actuals, predicted):
    cm = metrics.confusion_matrix(actuals, predicted, [1,0])
    sns.heatmap(cm, annot = True, fmt = '.2f', xticklabels = ['Good Response', 'Bad Response'], yticklabels = ['Good Response', 'Bad Response'])

    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")

    plt.show()


# In[106]:


cm = metrics.confusion_matrix(y_test, y_pred)
cm


# In[107]:


confusion_matrix(y_test, y_pred)


# In[ ]:




