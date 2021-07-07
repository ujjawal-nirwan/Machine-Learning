#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
sns.set()


# In[2]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor


# In[3]:


data=pd.read_csv('winequality-white.csv',sep=';')


# In[4]:


data.head()


# In[5]:


data.columns


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


sns.distplot(data['quality']);


# In[9]:


sns.pairplot(data, hue='quality', height=1.5, corner=True);


# In[10]:


sns.jointplot(data=data, x='fixed acidity', y='quality');


# In[11]:


sns.jointplot(data=data, x='volatile acidity', y='quality');


# In[12]:


sns.jointplot(data=data, x='citric acid', y='quality');


# In[13]:


sns.jointplot(data=data, x='residual sugar', y='quality');


# In[14]:


sns.jointplot(data=data, x='chlorides', y='quality');


# In[15]:


sns.jointplot(data=data, x='free sulfur dioxide', y='quality');


# In[16]:


sns.jointplot(data=data, x='total sulfur dioxide', y='quality');


# In[17]:


sns.jointplot(data=data, x='density', y='quality');


# In[18]:


sns.jointplot(data=data, x='pH', y='quality');


# In[19]:


sns.jointplot(data=data, x='sulphates', y='quality');


# In[20]:


sns.jointplot(data=data, x='alcohol', y='quality');


# ## Linear Regression

# In[21]:


y=data['quality']
X=data.copy()
X.drop('quality', axis=1, inplace=True)
x=sm.add_constant(X)


# In[22]:


x_features = data.columns


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


scaler = StandardScaler()


# In[25]:


scaler.fit(x)


# In[26]:


x_scaled = scaler.transform(x)


# In[27]:


x_scaled


# In[28]:


y_scale=((y-y.mean())/y.std())


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scale, test_size=0.7, random_state=17)


# In[30]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[31]:


y_hat = reg.predict(x_train)


# In[32]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.show()


# In[33]:


sns.distplot(y_train - y_hat)
plt.title("Residuals PDF", size=18);


# In[34]:


reg.score(x_train,y_train)


# In[35]:


reg.intercept_


# In[36]:


reg.coef_


# In[37]:


y_train_pred = reg.predict(x_train)
y_test_pred = reg.predict(x_test)
print('Mean squared error (train): {:.3f}'.format(mean_squared_error(y_train, y_train_pred)))
print('Mean squared error (holdout): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))


# In[38]:


reg_summary = pd.DataFrame(x.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[39]:


plt.figure(figsize=(10,10))
sns.barplot(x='Weights',y='Features', data=reg_summary);


# In[40]:


from sklearn import metrics


# In[41]:


y_test_pred=reg.predict(x_test)


# In[42]:


round(np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)),2)


# In[43]:


def get_train_test_rmse (model):
    y_test_pred_scale = model.predict(x_test)
    rmse_test = round(np.sqrt(metrics.mean_squared_error(y_test,y_test_pred_scale)),2)
    y_train_pred_scale = model.predict(x_train)
    rmse_train = round(np.sqrt(metrics.mean_squared_error(y_train,y_train_pred_scale)),2)
    print('train_rmse', rmse_train, 'test_rmse',rmse_test)


# In[44]:


get_train_test_rmse(reg)


# In[45]:


influence = pd.DataFrame(abs(reg.coef_),
                           index=x.columns,
                           columns=['influence'])
influence.sort_values('influence').plot(kind='barh');


# ## Apply Regularization

# In[46]:


from sklearn.linear_model import Ridge


# In[47]:


wine_ridge = Ridge (alpha = 1)


# In[48]:


wine_ridge.fit(x_train,y_train)


# In[49]:


get_train_test_rmse(wine_ridge)


# In[50]:


wine_ridge2 = Ridge (alpha=2)


# In[51]:


wine_ridge2.fit(x_train,y_train)


# In[52]:


get_train_test_rmse(wine_ridge2)


# In[53]:


wine_ridge3=Ridge(alpha=3)


# In[54]:


wine_ridge3.fit(x_train,y_train)


# In[55]:


get_train_test_rmse(wine_ridge3)


# ## LASSO Regression

# In[56]:


from sklearn.linear_model import Lasso


# In[57]:


lasso1=Lasso(alpha=0.01)


# In[58]:


lasso1.fit(x_train,y_train)


# In[59]:


get_train_test_rmse(lasso1)


# In[60]:


lasso1.coef_


# In[61]:


wine_ridge3.coef_


# In[62]:


columns_coef_df_lasso=pd.DataFrame(({'Columns' : data.columns, 'Coeff':lasso1.coef_}))


# In[63]:


columns_coef_df_lasso[columns_coef_df_lasso.Coeff == 0]


# In[64]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 17)


# In[67]:


importances = pd.DataFrame(abs(lasso1.coef_), index=x.columns, columns=['Importance'])
importances.sort_values('Importance').plot(kind='barh')


# ## LASSO CV

# In[69]:


alphas =[0.01,0.001,0.1,0.2,0.02,0.002] 
print('[', min(alphas), max(alphas), ']')

lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=17, n_jobs=-1)
lasso_cv.fit(x_train, y_train)
lasso_cv.alpha_


# In[70]:


importances = pd.DataFrame(abs(lasso_cv.coef_), index=x.columns, columns=['Importance'])
importances.sort_values('Importance').plot(kind='barh')


# In[71]:


y_train_pred = lasso_cv.predict(x_train)
y_test_pred = lasso_cv.predict(x_test)
print('Mean squared error (train): {:.3f}'.format(mean_squared_error(y_train, y_train_pred)))
print('Mean squared error (holdout): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))


# ## Random Classifier

# In[73]:


forest = RandomForestRegressor(n_estimators=10, random_state=17)
forest.fit(x_train, y_train)


# In[75]:


y_train_pred = forest.predict(x_train)
y_test_pred = forest.predict(x_test)
print('Mean squared error (train): {:.3f}'.format(mean_squared_error(y_train, y_train_pred)))
print('Mean squared error (holdout): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))


# In[78]:


forest_params = {'max_depth': range(10, 25), 'max_features': range(6,12)}

locally_best_forest = GridSearchCV(forest, forest_params,scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
locally_best_forest.fit(x_train, y_train)
print(abs(locally_best_forest.best_score_), locally_best_forest.best_params_)


# In[80]:


importances = pd.DataFrame(locally_best_forest.best_estimator_.feature_importances_,index=x.columns,columns=['Importance'])
importances.sort_values('Importance').plot(kind='barh');


# In[ ]:




