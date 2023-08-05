#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd   #most important thing is what we need to find is alwasys y 
                      # and x is dependent veriable we use the x for find the c and y
                    #machine learning is depend no y=b+cx is


# In[2]:


dataset = pd.read_csv("50_Startups.csv")


# In[3]:


dataset.info()


# In[4]:


dataset  #head is function use for show the top data


# In[5]:


dataset.head(10)


# In[6]:


dataset.shape   #shape is use to show the how many row and columns


# In[7]:


dataset.columns    #this step show the columns


# In[8]:


y = dataset['Profit']    #y is our dependent variable


# In[9]:


y


# In[10]:


x = dataset[['R&D Spend',  
          'Administration',
           'Marketing Spend',
           'State']]                #x is our independent variable means all the x value is decide the value of y 


# In[11]:


x


# In[12]:


state = dataset["State"]   #state is the Categorical variable


# In[13]:


pd.get_dummies(state)    #one hot encoding method   


# In[14]:


state_dummy = pd.get_dummies(state)    #one hot encoding method


# In[15]:


type(state_dummy)


# In[25]:


final_dummy_variable = state_dummy.iloc[  : , 0:2]  #dummy variable trap


# In[26]:


final_dummy_variable


# In[27]:


y = dataset['Profit']


# In[28]:


x = dataset[['R&D Spend',
          'Administration',
           'Marketing Spend']]


# In[29]:


x


# In[38]:


X = pd.concat([x , final_dummy_variable],axis=1)


# In[39]:


X


# In[40]:


X.shape


# In[41]:


y.shape


# In[42]:


from sklearn.model_selection import train_test_split   


# In[43]:


X_train, X_test, y_train , y_test= train_test_split(X,y ,test_size=0.20 , random_state=1)


# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


model = LinearRegression()


# In[46]:


model


# In[47]:


model.fit(X_train,y_train)


# In[48]:


y_preadict = model.predict(X_test)


# In[49]:


y_preadict


# In[50]:


X_test


# In[51]:


y_test


# In[52]:


model.fit(X,y)


# In[53]:


from sklearn import metrics


# In[54]:


metrics.mean_absolute_error(y_test,y_preadict)


# In[55]:


model.coef_


# In[56]:


model.intercept_


# In[ ]:





# In[ ]:





# In[ ]:




