#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder

liver_df=pd.read_csv(r'E:\Data\liver.csv')


# In[3]:


liver_df.isnull().sum()


# In[4]:


liver_df['Albumin_and_Globulin_Ratio'].mean()
liver_df=liver_df.fillna(0.94)


# In[5]:


sns.countplot(data=liver_df, x = 'Dataset', label='Count')

LD, NLD = liver_df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[6]:


sns.countplot(data=liver_df, x = 'Gender', label='Count')

M, F = liver_df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)


# In[7]:


sns.catplot(x="Age", y="Gender", hue="Dataset", data=liver_df);


# In[8]:


sns.catplot(x="Dataset", y="Albumin", data=liver_df);


# In[9]:


X = liver_df.drop(['Gender','Dataset'], axis=1)
X.head(3)


# In[10]:


#now lets split the dataset into train and test into 70-30 ratio
from sklearn.model_selection import train_test_split
X=liver_df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=liver_df['Dataset']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

 
# Creating logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




