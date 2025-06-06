import requests
import zipfile
import io
import pandas as pd#creates nd arrays with much faster implementation than lists
import numpy as np#loading datasets,data manipulation etc
from sklearn.model_selection import train_test_split#
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
response=requests.get(zip_url)
if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        with thezip.open('student-por.csv') as myfile:
            df_por = pd.read_csv(myfile, sep=';')#conversion of csv into dataframe
        with thezip.open('student-mat.csv') as myfile:
            df_mat = pd.read_csv(myfile, sep=';')
    df = pd.concat([df_por, df_mat], axis=0)
else:
    print("Failed to download the file")

print(df.head())
print(df.describe())
print(df.isnull().sum())#isnull checks if there is any empty cell and sum accumulates all true values add it in one coloumn
print(df.info())
#to check the pass percentage i gonna convert the g3 data into classification problem
#with 40% ie 8 marks
df['G3']=(df['G3']>=8).astype(int)
#handling missing values
df.fillna(df.mean(numeric_only=True),inplace=True)
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].fillna(df[column].mode()[0])

#encoding
#use label encoder on any categorical variable and the method fit_transform in this does the job
# Encode binary categorical variables using LabelEncoder
binary_vars = ['school', 'sex', 'address', 'famsize', 'Pstatus',
               'schoolsup', 'famsup', 'paid', 'activities',
               'nursery', 'higher', 'internet', 'romantic']
for var in binary_vars:
    le = LabelEncoder()
    df[var] = le.fit_transform(df[var])
# Encode nominal categorical variables using get_dummies
nominal_vars = ['Mjob', 'Fjob', 'reason', 'guardian']

df = pd.get_dummies(df, columns=nominal_vars, drop_first=True)
print(df.select_dtypes(include=['object']).columns)
# Normalize numeric values
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('G3')

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('G3')
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
# Preparing the data
X = df.drop(['G3'], axis=1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.select_dtypes(include=['object']).columns)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Model training using Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

from sklearn.metrics import confusion_matrix
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
