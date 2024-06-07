from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
plt.ion()

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("onlinefraud.csv")

df_lr_model = df.copy()

# Normalizing the transaction types to better train the data
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df_cat = df_lr_model['type'].values
df_cat_encoded = encoder.fit_transform(df_cat)
df_lr_model['type'] = encoder.fit_transform(df_lr_model['type'])
df_lr_model = df_lr_model.drop(columns=['nameOrig', 'nameDest'])

X = df_lr_model.drop(columns=['isFraud'], axis=1)
y = df_lr_model['isFraud'].copy()

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

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

dump(log_model, 'lr_model.joblib')
dump(scaler, 'scaler.joblib')
dump(y_test, 'y_test.joblib')
dump(X_test_scaled, 'X_test_scaled.joblib')
dump(encoder, 'lr_encoder.joblib')