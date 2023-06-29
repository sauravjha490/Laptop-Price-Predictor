#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[2]:


laptop_df = pd.read_csv('laptop_data.csv')
laptop_df.head(3)


# In[3]:


laptop_df.shape


# In[4]:


laptop_df.info()


# In[5]:


laptop_df['RAM'] = laptop_df['RAM'].str.replace('GB','')
laptop_df['Weight'] = laptop_df['Weight'].str.replace('kg','')
laptop_df['RAM'] = laptop_df['RAM'].astype('int32')
laptop_df['Weight'] = laptop_df['Weight'].astype('float32')


# In[6]:


laptop_df.info()


# # EDA Analysis:

# ### Distribution of Target column:

# In[7]:


sns.displot(laptop_df['Price'],kind='kde')
plt.show()


# ### Distribution of Company column:

# In[8]:


laptop_df['Company'].value_counts().plot(kind='bar')
plt.show()


# In[9]:


# How 'Laptop Brand' affects the price of Laptop:
plt.figure(figsize=(18,6))
sns.barplot(x=laptop_df['Company'],y=laptop_df['Price'])
plt.show()


# ### Types of Laptops:-

# In[10]:


laptop_df['TypeName'].value_counts().plot(kind='bar')
plt.show()


# In[11]:


# laptop_df['TypeName'].value_counts().plot(kind='bar') 
plt.figure(figsize=(10,6))
sns.barplot(x=laptop_df['TypeName'],y=laptop_df['Price'])
plt.show()


# - Majority people prefer 'Notebook' as it is in budget of the buyer.

# In[12]:


# Does Price Vary with Laptop size in 'Inches':

sns.scatterplot(x=laptop_df['Inches'],y=laptop_df['Price'])
plt.show()


# - From the above plot we can conclude that there is a relationship but not a strong relationship between the price and size column.

# # Extract Touchscreen Information:

# In[13]:


laptop_df['Touch_Screen'] = laptop_df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

#sns.countplot(laptop_df['Touch_Screen'])

sns.barplot(x=laptop_df['Touch_Screen'],y=laptop_df['Price'])
plt.show()


# # Extract IPS Channel presence Information:

# In[14]:


laptop_df['IPS'] = laptop_df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
#sns.countplot(laptop_df['IPS'])

sns.barplot(x=laptop_df['IPS'],y=laptop_df['Price'])
plt.show()


# In[15]:


def X_resolution(s):
    return s.split()[-1].split("x")[0]
def Y_resolution(s):
    return s.split()[-1].split("x")[1]

laptop_df['X_res'] = laptop_df['ScreenResolution'].apply(lambda x:X_resolution(x))
laptop_df['Y_res'] = laptop_df['ScreenResolution'].apply(lambda y:Y_resolution(y))

laptop_df['X_res'] = laptop_df['X_res'].astype('int')
laptop_df['Y_res'] = laptop_df['Y_res'].astype('int')


# In[ ]:





# In[16]:


laptop_df['ppi'] = (((laptop_df['X_res']**2) + (laptop_df['Y_res']**2))**0.5/laptop_df['Inches']).astype('float')


# In[17]:


laptop_df.corr()['Price'].sort_values(ascending=False)


# In[18]:


laptop_df.drop(columns=['ScreenResolution', 'Inches','X_res','Y_res'],inplace=True)


# In[19]:


laptop_df.head(2)


# # CPU Column:

# In[20]:


laptop_df['CPU'].value_counts().head()


# - If you observe the CPU column then it also contains lots of information. If you again use a unique function or value counts function on the CPU column then we have 118 different categories. The information it gives is about preprocessors in laptops and speed.

# In[21]:


def feth_processor(x):
    cpu_name = " ".join(x.split()[:3])
    
    if cpu_name == 'Intel Core i7' or cpu_name == 'Intel Core i5' or cpu_name == 'Intel Core i3':
        return cpu_name
    elif cpu_name.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'

laptop_df['CPU_Brand'] = laptop_df['CPU'].apply(lambda x:feth_processor(x))   


# In[22]:


sns.barplot(x=laptop_df['CPU_Brand'],y=laptop_df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[23]:


laptop_df.drop(columns='CPU',inplace=True)


# In[24]:


sns.barplot(x=laptop_df['RAM'], y=laptop_df['Price'])
plt.show()


# In[25]:


laptop_df.head(2)


# # Memory Column:

# In[26]:


laptop_df['Memory'].unique()


# In[27]:


a = '256GB SSD +  1.0TB Hybrid'
new_1 = a.split('+')
new_1[1]


# In[28]:


laptop_df['Memory'] = laptop_df['Memory'].astype('str').replace('.0','',regex=True)
laptop_df['Memory'] = laptop_df['Memory'].str.replace('GB','')
laptop_df['Memory'] = laptop_df['Memory'].str.replace('TB','000')
new = laptop_df['Memory'].str.split("+",n=1,expand=True)

laptop_df['first'] = new[0]
laptop_df['first']  = laptop_df['first'].str.strip()
laptop_df['second'] = new[1]

laptop_df['Layer1HDD'] = laptop_df['first'].apply(lambda x:1 if "HDD" in x else 0)
laptop_df['Layer1SSD'] = laptop_df['first'].apply(lambda x:1 if "SSD" in x else 0)
laptop_df['Layer1Hybrid'] = laptop_df['first'].apply(lambda x:1 if "Hybrid" in x else 0)
laptop_df['Layer1Flash_Storage'] = laptop_df['first'].apply(lambda x:1 if "Flash Storage" in x else 0)
laptop_df['first'] = laptop_df['first'].str.replace(r'\D','')

laptop_df['second'].fillna('0',inplace=True)
laptop_df['Layer2HDD'] = laptop_df['second'].apply(lambda x:1 if "HDD" in x else 0)
laptop_df['Layer2SSD'] = laptop_df['second'].apply(lambda x:1 if "SSD" in x else 0)
laptop_df['Layer2Hybrid'] = laptop_df['second'].apply(lambda x:1 if "Hybrid" in x else 0)
laptop_df['Layer2Flash_Storage'] = laptop_df['second'].apply(lambda x:1 if "Flash Storage" in x else 0)
laptop_df['second'] = laptop_df['second'].str.replace(r'\D','')

laptop_df['first'] = laptop_df['first'].apply(lambda x: int(x) if x else None)
laptop_df['second'] = laptop_df['second'].apply(lambda x: int(x) if x else None)

laptop_df["HDD"]=(laptop_df["first"]*laptop_df["Layer1HDD"]+laptop_df["second"]*laptop_df["Layer2HDD"])
laptop_df["SSD"]=(laptop_df["first"]*laptop_df["Layer1SSD"]+laptop_df["second"]*laptop_df["Layer2SSD"])
laptop_df["Hybrid"]=(laptop_df["first"]*laptop_df["Layer1Hybrid"]+laptop_df["second"]*laptop_df["Layer2Hybrid"])
laptop_df["Flash_Storage"]=(laptop_df["first"]*laptop_df["Layer1Flash_Storage"]+laptop_df["second"]*laptop_df["Layer2Flash_Storage"])


laptop_df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[29]:


laptop_df.drop(columns='Memory',inplace=True)


# In[30]:


laptop_df.head(2)


# In[31]:


# GPU Variable:

laptop_df['GPU_Brand'] = laptop_df['GPU'].apply(lambda x: x.split()[0])
laptop_df.drop(columns='GPU',inplace=True)
laptop_df = laptop_df[laptop_df['GPU_Brand']!='ARM']
laptop_df.head(2)


# In[32]:


sns.barplot(x=laptop_df['GPU_Brand'],y=laptop_df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# # Operating System Column:

# In[33]:


sns.barplot(x=laptop_df['OperatingSystem'],y=laptop_df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[34]:


laptop_df['OperatingSystem'].value_counts()


# In[35]:


laptop_df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[36]:


def Operating_Sys(inp):
    if inp=='Windows 10' or inp=='Windows 7' or inp=='Windows 10 S':
        return 'Windows'
    elif inp=='macOS' or inp== 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[37]:


laptop_df['OS'] = laptop_df['OperatingSystem'].apply(Operating_Sys)


# In[38]:


laptop_df.drop(columns='OperatingSystem',inplace=True)
laptop_df.head(2)


# In[39]:


sns.barplot(x=laptop_df['OS'],y=laptop_df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[40]:


sns.distplot(np.log(laptop_df['Price']))
plt.show()


# In[41]:


laptop_df.head(2)


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error


# In[43]:


laptop_df.head()


# In[44]:


laptop_df.info()


# In[45]:


laptop_df['HDD'].fillna(0,inplace=True)
laptop_df['SSD'].fillna(0,inplace=True)


# In[46]:


X = laptop_df.drop(columns=['Price'])
Y = np.log(laptop_df['Price'])


# In[47]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=42)


# In[48]:


X.head(2)


# ## Regression Model:

# In[49]:


# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
# step2 = LinearRegression()

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,Y_train)
# Y_pred_reg = pipe.predict(X_test)
# print('R2 Score: ',r2_score(Y_test,Y_pred_reg))
# print('Mean_absolute_error: ',mean_absolute_error(Y_test,Y_pred_reg))


# ## KNN Model:

# In[57]:


# # Rebuilding model according to best k value = 3

# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
# step2 = KNeighborsRegressor(n_neighbors=3)

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,Y_train)
# Y_pred_KNN = pipe.predict(X_test)
# print('R2 Score: ',r2_score(Y_test,Y_pred_KNN))
# print('Mean_absolute_error: ',mean_absolute_error(Y_test,Y_pred_KNN))


# ## SVM Model:

# In[52]:


# from sklearn.svm import SVR
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
# step2 = SVR(kernel='rbf',epsilon=0.1,C=10000)

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,Y_train)
# Y_pred_svr = pipe.predict(X_test)
# print('R2 Score: ',r2_score(Y_test,Y_pred_svr))
# print('Mean_absolute_error: ',mean_absolute_error(Y_test,Y_pred_svr))


# ## Decision Tree Model:

# In[53]:


# from sklearn.tree import DecisionTreeRegressor
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
# step2 = DecisionTreeRegressor(max_depth=8,min_samples_split=2,min_samples_leaf=1)

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,Y_train)
# Y_pred_dt = pipe.predict(X_test)
# print('R2 Score: ',r2_score(Y_test,Y_pred_dt))
# print('Mean_absolute_error: ',mean_absolute_error(Y_test,Y_pred_dt))


# ## Random Forest Model:

# In[54]:


# from sklearn.ensemble import RandomForestRegressor
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
# step2 = RandomForestRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,Y_train)
# Y_pred_rfr = pipe.predict(X_test)
# print('R2 Score: ',r2_score(Y_test,Y_pred_rfr))
# print('Mean_absolute_error: ',mean_absolute_error(Y_test,Y_pred_rfr))


# ## XgBoost Model:

# In[55]:


# from xgboost import XGBRegressor
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
# step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,Y_train)
# Y_pred_xg = pipe.predict(X_test)
# print('R2 Score: ',r2_score(Y_test,Y_pred_xg))
# print('Mean_absolute_error: ',mean_absolute_error(Y_test,Y_pred_xg))


# # Modeling without using Pipeline:

# In[67]:


from sklearn.preprocessing import LabelEncoder
X_obj = X.select_dtypes(include='object')
encoder = LabelEncoder()
X_obj_new = X_obj.apply(encoder.fit_transform)
X_obj_new.head(2)


# In[66]:


X_num = X.select_dtypes(include='number')
X_num.head(2)


# In[69]:


Final_df = pd.concat([X_num,X_obj_new],axis=1)
Final_df.head(2)


# In[70]:


X = Final_df
Y = Y = np.log(laptop_df['Price'])


# In[72]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=42)


# In[77]:


# Regression Model_Without Pipeline:
regress = LinearRegression()
regress.fit(X_train,Y_train)

# prediction:
Y_regress = regress.predict(X_test)
Y_regress[:5]

# accuracy:
r2_score(Y_test,Y_regress)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Exporting the Model:

# In[78]:


# Save the Model:
import pickle
pickle.dump(regress,open('model.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




