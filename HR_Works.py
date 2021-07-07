#!/usr/bin/env python
# coding: utf-8

# ### 1.  Problem Statement:
# - The problem states that 30% of the candidates who accept the job offer do not join the company. HRWorks wants to find if a model can be built to predict the likelihood of a candidate joining the company.<br>
# - Now we have to compare all the factors that tell us that what factors are affecting for a candidate so that they are not joining the company.<br>
# - We have to also relate these factors to understand whether a candidate will join or not based on the given circumstances in future.
# 
# ###  Why it is needed:
# 
# - As we know taking interviews and exams of applied candidates includes some kind of resources and time which may lead to loss if a candidate doesn't join after clearing all the stages.<br>
# - Because of this sometime we loose a candidate who is willing to work but scored less in eligibility criteria.<br>
# - Sometimes we loose a good or best candidate or we can say desired candidate that will help to increase the future of the company ot will increase the revenue of the company in some way. So we want to understand what the problem they are facing in joing the company.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
import sklearn
import statsmodels.api as sm 
import scipy.stats as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
sns.set()


# In[2]:


data=pd.read_csv("hr_data.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.describe().T


# As we can see that there are outliers present in Are which are needed in the analysis

# In[5]:


data.describe(include=object).T


# In[6]:


data.info()


# In[7]:


data.columns


# In[8]:


data.isnull().sum()


# In[9]:


data.duplicated().sum()


# In[10]:


data.dropna()


# ## Treating missing values

# As age contains outliers we will use <b> Median </b> instead of <b> Mean </b>

# In[11]:


data['Age'].fillna(data["Age"].median(),inplace=True)


# In[12]:


data['Offered band'].value_counts()


# In[13]:


imputer=data.copy()


# In[14]:


freq_imputer=SimpleImputer(strategy='most_frequent')


# In[15]:


imputer.loc[:,['Offered band']]=freq_imputer.fit_transform(imputer.loc[:,['Offered band']])


# In[16]:


data=imputer.copy()


# In[17]:


data.isnull().sum()


# ### 2. Hypothesis:
# 
# - H<sub>0</sub>: Based on the given information by a candidate will join
# - H<sub>1</sub>: Based on the given information by a candidate will not join

# ### 3. Data Exploration

# In[18]:


data.columns


# In[19]:


data['Status'].value_counts()


# In[20]:


sns.countplot(data['Status'])
plt.xticks(rotation=90)
plt.xlabel('These are Job Status');


# Most of the candidate have joined

# In[21]:


def countp(x, data):
    plt.figure(figsize = (10,8))
    sns.countplot(x = x, data = data, hue = 'Status', palette = 'rocket')
    plt.xlabel(x)
    plt.xticks(rotation=90)


# In[22]:


countp('Age',data)


# We can see that people falling under 34 years age are most joined and not joined

# In[23]:


countp('Gender',data)


# Males are most joined and not joined among genders

# In[24]:


countp('DOJ Extended',data)


# Those who got Date of Joining, extended joined and not joined slightly more 

# In[25]:


countp('Notice period',data)


# Those who got a notice period of 30 days joined and not joined more

# In[26]:


countp('Offered band',data)


# Most of the people joined and not joined E1 Offered band

# In[27]:


countp('Location',data)


# Most of the people joined are from Chennai and Noida while not joined are Chennai and Banglore

# In[28]:


countp('LOB',data)


# MOst of the people joined in INFRA and ERS Line of Bussiness while not joined in ERS, INFRA and BFSi

# In[29]:


countp('Joining Bonus',data)


# Those who didn't got any joining bonus are most who joined or not joined

# In[30]:


countp('Rex in Yrs',data)


# Those who got age relaxation between 3 to 4 yrs joined and not joined more

# In[31]:


countp('Candidate Source',data)


# Those who came direct are the most joined and not joined while those who are refered by company are least joined and not joined

# ### Treatment of outliers

# Removing Outliers from Age

# In[32]:


sns.distplot(data['Age']);


# In[33]:


data=data[data['Age'] < data['Age'].quantile(.99)]


# In[34]:


sns.distplot(data['Age']);


# Removing Outliers from Pecent hike expected in CT

# In[35]:


sns.distplot(data['Pecent hike expected in CTC']);


# In[36]:


data=data[data['Pecent hike expected in CTC'] < data['Pecent hike expected in CTC'].quantile(.99)]


# In[37]:


sns.distplot(data['Pecent hike expected in CTC']);


# Removing Outliers from Percent hike offered in CTC

# In[38]:


sns.distplot(data['Percent hike offered in CTC']);


# In[39]:


data=data[data['Percent hike offered in CTC'] < data['Percent hike offered in CTC'].quantile(.99)]


# In[40]:


sns.distplot(data['Percent hike offered in CTC']);


# Removing Outliers from Percent difference CTC

# In[41]:


sns.distplot(data['Percent difference CTC']);


# In[42]:


data=data[data['Percent difference CTC'] < data['Percent difference CTC'].quantile(.99)]


# In[43]:


sns.distplot(data['Percent difference CTC']);


# Removing Outliers from Rex in Yrs

# In[44]:


sns.distplot(data['Rex in Yrs']);


# In[45]:


data=data[data['Rex in Yrs'] < data['Rex in Yrs'].quantile(.99)]


# In[46]:


sns.distplot(data['Rex in Yrs']);


# ## 4. Model

# ### Logistic Regression

# In[47]:


data['Status']=data['Status'].apply(lambda x:0 if x=='Not Joined' else 1)


# In[48]:


x_feature = list(data.columns)
x_feature.remove('Status') 
x_feature.remove("Pecent hike expected in CTC") 
x_feature.remove("Percent hike offered in CTC")
x_feature.remove("SLNO")
x_feature.remove("Candidate Ref")


# In[49]:


encoded_data = pd.get_dummies(data[x_feature],drop_first=True)


# In[50]:


y = data['Status']
x1 = encoded_data
x = sm.add_constant(x1)


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.8, random_state = 42)


# In[52]:


logit = sm.Logit(y_train, x_train)


# In[53]:


logit=logit.fit(method='bfgs')


# In[54]:


logit.summary2()


# In[55]:


y_pred = logit.predict(x_test)
y_pred


# In[56]:


def get_significant_variables(lm):
    var_p_vals = pd.DataFrame(lm.pvalues)
    var_p_vals['vars'] = var_p_vals.index
    var_p_vals.columns = ['pvals','vars']
    return list(var_p_vals[var_p_vals.pvals <= 0.05]['vars'])


# In[57]:


significant_variables = get_significant_variables(logit)
significant_variables


# In[58]:


plt.figure(figsize=(20,10))
sns.heatmap(encoded_data[significant_variables].corr(), annot = True, fmt ='.1f')


# In[59]:


log_reg = LogisticRegression(solver = 'lbfgs', max_iter=100)


# In[60]:


log_reg.fit(x_train, y_train)


# In[61]:


y_pred = log_reg.predict(x_test)
y_pred[0:100]


# In[62]:


log_reg.predict_proba(x_test)[0:28]


# In[63]:


log_reg.coef_[0:100]


# In[64]:


def confusion_matrix(actuals,predicted):
    cm = metrics.confusion_matrix(actuals,predicted,[1,0])
    sns.heatmap(cm, annot = True, fmt ='.2f', xticklabels=['Joined', 'Not Joined'], 
                yticklabels=['Joined', 'Not Joined'])
    
    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")


# In[65]:


print(metrics.classification_report(y_test, log_reg.predict(x_test)))


# In[66]:


confusion_matrix(y_test, y_pred)


# ### Decision Tree Classifier

# In[67]:


dtc = DecisionTreeClassifier()


# In[68]:


dtc.fit(x_train,y_train)


# In[69]:


y_pred_dt = dtc.predict(x_test)


# In[70]:


y_pred_dt_train = dtc.predict(x_train)


# In[71]:


print(metrics.classification_report(y_test, y_pred_dt))


# In[72]:


confusion_matrix(y_test, y_pred_dt)


# ### KNeighbors Classifier

# In[73]:


knn_classifier = KNeighborsClassifier()


# In[74]:


knn_classifier.fit(x_train,y_train)


# In[75]:


y_pred_knn=knn_classifier.predict(x_test)


# In[76]:


print(classification_report(y_test, y_pred))


# In[77]:


confusion_matrix(y_test, y_pred_knn)


# ### Random Forest Regressor

# In[78]:


forest = RandomForestRegressor(n_estimators=10, random_state=17)
forest.fit(x_train, y_train)


# In[79]:


y_train_pred = forest.predict(x_train)
y_test_pred = forest.predict(x_test)
print('Mean squared error (train): {:.3f}'.format(mean_squared_error(y_train, y_train_pred)))
print('Mean squared error (holdout): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))


# In[80]:


forest_params = {'max_depth': range(10, 25), 'max_features': range(6,12)}
locally_best_forest = GridSearchCV(forest, forest_params,scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
locally_best_forest.fit(x_train, y_train)
print(abs(locally_best_forest.best_score_), locally_best_forest.best_params_)


# In[81]:


importances = pd.DataFrame(locally_best_forest.best_estimator_.feature_importances_,index=x.columns,columns=['Importance'])
importances.sort_values('Importance').plot(kind='barh', figsize = (10,8));


# #### 4.Comparing different models.
# Out of Logistic Regression(Accuracy= 0.82, True positive values= 1193), Decision Tree Classifier(Accuracy= 0.74, True positive values=1251),), KNeighbors Classifier(Accuracy= 0.82, True positive values=1337), Logistic Regression and KNeighbors Classifier have same accuracy but KNeighbors Classifier have more true predicted values. There out of three KNeighbors Classifier model is best suited for this problem
