#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
liv_df=pd.read_csv(r'E:\liver.csv')


# In[78]:


liv_df.isnull().sum()


# In[79]:


g = sns.FacetGrid(liv_df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="blue")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[80]:



liv_df['Albumin_and_Globulin_Ratio'].mean()
liv_df=liv_df.fillna(0.94)


# In[81]:


liv_df.isnull().sum()


# In[82]:


from sklearn.model_selection import train_test_split
#Drop_gender = liv_df.drop(labels=['Gender' ],axis=1 )

#X = Drop_gender
liv_df['Gender']=liv_df['Gender'].apply(lambda x:1 if x=='Male' else 0)


X=liv_df[['Age','Gender','Total_Bilirubin', 'Direct_Bilirubin',
         'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
        'Albumin_and_Globulin_Ratio']]
y = liv_df['Dataset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)


# In[83]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[84]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
rf_predicted = random_forest.predict(X_test)
random_forest_score_train = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)

print('Random Forest Training Score: \n', random_forest_score_train)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))


# In[85]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
svm_predicted = svclassifier.predict(X_test)
svc_score_train = round(svclassifier.score(X_train, y_train) * 100, 2)
svc_score_test = round(svclassifier.score(X_test, y_test) * 100, 2)

print('SVM Training Score: \n', svc_score_train)
print('SVM Test Score: \n', svc_score_test)
print('Accuracy: \n', accuracy_score(y_test,svm_predicted))




