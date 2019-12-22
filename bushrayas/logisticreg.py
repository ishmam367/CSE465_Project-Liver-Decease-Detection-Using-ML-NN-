import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("indian_liver_patient.csv")
df.head()
df.info()
df['Dataset'].unique()
df.describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df.head()
df['Albumin_and_Globulin_Ratio'].isna().sum() 
df['Albumin_and_Globulin_Ratio'].hist()
df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace = True)
corr = df.corr()
sns.heatmap(corr)
X = df.drop(['Dataset'], axis = 1)

Y = df['Dataset']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_pred, y_test))
sns.countplot(df['Dataset'])