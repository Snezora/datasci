from joblib import dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
plt.ion()

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("onlinefraud.csv")

df_dt_model = df.copy()

# Normalizing the transaction types to better train the data
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df_cat = df_dt_model['type'].values
df_cat_encoded = encoder.fit_transform(df_cat)
df_dt_model['type'] = encoder.fit_transform(df_dt_model['type'])
df_dt_model = df_dt_model.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

X = df_dt_model.drop(columns=['isFraud'], axis=1)
y = df_dt_model['isFraud'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Standardization (Z-Score Normalization) 
# StandardScaler for Data
# Reference: https://www.kaggle.com/code/georgehanymilad/online-payments-fraud-detection

scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Transform the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the model with the optimal max_depth, 19
dt_model = DecisionTreeClassifier(max_depth=19, random_state=1, max_features='sqrt', min_samples_leaf=4, min_samples_split=3, criterion='gini')
dt_model.fit(X_train_scaled, y_train)

dump(dt_model, 'dt_model.joblib')
dump(encoder, 'dt_encoder.joblib')
