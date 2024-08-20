#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
df1=pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
df2=pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")


# In[2]:


data = pd.merge(df1, df2)
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.drop('Code',axis=1,inplace=True)


# In[5]:


data.head()


# In[6]:


data.size,data.shape


# In[7]:


data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)


# In[8]:


data.head()


# In[9]:


plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Blues')
plt.plot()


# In[10]:


sns.jointplot(data,x='Schizophrenia',y='mental_fitness',kind='reg',color='m')
plt.show()


# In[11]:


sns.jointplot(data,x='Bipolar_disorder',y='mental_fitness',kind='reg',color='blue')
plt.show()


# In[12]:


sns.pairplot(data,corner=True)
plt.show()


# In[13]:


mean = data['mental_fitness'].mean()
mean


# In[14]:


fig = px.pie(data, values='mental_fitness', names='Year')
fig.show()


# In[15]:


fig = px.line(data, x="Year", y="mental_fitness", color='Country',markers=True,color_discrete_sequence=['red','blue'],template='plotly_dark')
fig.show()


# In[16]:


df = data.copy()
df.head()


# In[17]:


df.info()


# In[18]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])


# In[19]:


X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)


# In[20]:


from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create an Elastic Net model with a specific alpha (regularization strength) and l1_ratio
# The l1_ratio controls the balance between L1 and L2 regularization (1.0 for Lasso, 0.0 for Ridge).
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net.fit(xtrain, ytrain)

# Predict on the training set
ytrain_pred = elastic_net.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = np.sqrt(mse)
r2 = r2_score(ytrain, ytrain_pred)
trelasti_mse = mse
trelasti_rmse = rmse
trelasti_r2 = r2
print("The model performance for the training set using ElasticNet")
print("\n")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# Predict on the testing set
ytest_pred = elastic_net.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ytest_pred)
elasti_mse = mse
elasti_rmse = rmse
elasti_r2 = r2

print("The model performance for the testing set using ElasticNet Regression")
print("\n")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[21]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create a KNN regression model with your desired parameters
knn = KNeighborsRegressor(n_neighbors=5)  # You can adjust n_neighbors and other hyperparameters

# Train the KNN model on the training data
knn.fit(xtrain, ytrain)

# Predict on the training set
ytrain_pred = knn.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = np.sqrt(mse)
r2 = r2_score(ytrain, ytrain_pred)
trknn_mse = mse
trknn_rmse = rmse
trknn_r2 = r2

print("The model performance for the training set using KNN Algorithm")
print("\n")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# Predict on the testing set
ytest_pred = knn.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ytest_pred)
knn_mse = mse
knn_rmse = rmse
knn_r2 = r2

print("The model performance for the testing set using KNN Algorithm")
print("\n")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[22]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create a Random Forest regression model with your desired parameters
random_forest = RandomForestRegressor(n_estimators=100, random_state=0)  # You can adjust n_estimators and other hyperparameters

# Train the Random Forest model on the training data
random_forest.fit(xtrain, ytrain)

# Predict on the training set
ytrain_pred = random_forest.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = np.sqrt(mse)
r2 = r2_score(ytrain, ytrain_pred)
trrando_mse = mse
trrando_rmse = rmse
trrando_r2 = r2


print("The model performance for the training set using Random Forest Algorithm")
print("\n")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# Predict on the testing set
ytest_pred = random_forest.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ytest_pred)
rando_mse = mse
rando_rmse = rmse
rando_r2 = r2
print("The model performance for the testing set using Random Forest Algorithm")
print("\n")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:





# In[27]:


import matplotlib.pyplot as plt

# Assuming the variables are already defined
# elasti_mse = mse
# elasti_rmse = rmse
# elasti_r2 = r2
# rando_mse = mse
# rando_rmse = rmse
# rando_r2 = r2
# knn_mse = mse
# knn_rmse = rmse
# knn_r2 = r2

# Metrics for models
metrics = ['MSE', 'RMSE', 'R2']
elasti_values = [elasti_mse, elasti_rmse, elasti_r2]
rando_values = [rando_mse, rando_rmse, rando_r2]
knn_values = [knn_mse, knn_rmse, knn_r2]
x = range(len(metrics))

# Creating the bar plot
bar_width = 0.2  # Width of each bar

# Calculate the x positions for each set of bars
elasti_x = [x[i] - bar_width for i in range(len(metrics))]
rando_x = [x[i] for i in range(len(metrics))]
knn_x = [x[i] + bar_width for i in range(len(metrics))]

plt.bar(elasti_x, elasti_values, width=bar_width, label='ElasticNet', color='b', align='center')
plt.bar(rando_x, rando_values, width=bar_width, label='RandomForest', color='r', align='edge')
plt.bar(knn_x, knn_values, width=bar_width, label='KNN', color='g', align='edge')

# Adding labels on top of the bars with 3 decimal places
for i in range(len(metrics)):
    plt.text(elasti_x[i], elasti_values[i], f'{elasti_values[i]:.3f}', ha='center', va='bottom')
    plt.text(rando_x[i], rando_values[i], f'{rando_values[i]:.3f}', ha='center', va='bottom')
    plt.text(knn_x[i], knn_values[i], f'{knn_values[i]:.3f}', ha='center', va='bottom')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Comparison of ElasticNet, RandomForest Metrics and KNN Metrics')
plt.xticks(x, metrics)
plt.legend()
plt.show()


# In[ ]:




