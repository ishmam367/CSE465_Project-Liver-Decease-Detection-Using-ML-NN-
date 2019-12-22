import numpy as np 
import pandas as pd 
filepath = 'indian_liver_patient.csv'
data = pd.read_csv(filepath)
data.head()
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

X = data.iloc[:,:-1].values
t = data.iloc[:,-1].values
for u in range(len(t)):
    if t[u] == 2:
        t[u] = 0
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
X[:,1] = lbl.fit_transform(X[:,1])
data.isnull().any()
data['Albumin_and_Globulin_Ratio'].isnull().sum()
missing_values_rows = data[data.isnull().any(axis=1)]
print(missing_values_rows)
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:,9:10] = imp.fit_transform(X[:,9:10])
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X,t,random_state=0,test_size=0.20)
from sklearn. preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,2:] = sc.fit_transform(X_train[:,2:])
X_test[:,2:] = sc.transform(X_test[:,2:])
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
# creating object of SVC class
classifier_svc = SVC(kernel='rbf', random_state=0, gamma='auto')
# fitting the model/ training the model on training data (X_train,t_train)
classifier_svc.fit(X_train,t_train)
# predicting whether the points (people/rows) in the test set (X_test) have the liver disease or not
y_pred_svc = classifier_svc.predict(X_test)
# evaluating model performance by confusion-matrix
cm_svc = confusion_matrix(t_test,y_pred_svc)
print(cm_svc)
# accuracy-result of SVC model
accuracy_svc = accuracy_score(t_test,y_pred_svc)
print('The accuracy of SupportVectorClassification is : ', str(accuracy_svc*100) , '%')