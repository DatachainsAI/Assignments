import pandas as pd            # Data manipulation
import numpy as np             # Numerical operations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('transactions.csv')

print("[3] Class Distribution:") #class distribution
print(df['Fraudulent'].value_counts())

#feature selection
features = ['Amount', 'MerchantCategory', 'PaymentMethod', 'NewDeviceFlag', 'Country', 'AvgTxSize', 'TxFrequency']
X = pd.get_dummies(df[features], drop_first=True)  # Categorical one-hot encoding
y = df['Fraudulent']

#feature scaling - mean and variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=25, stratify=y)


smote = SMOTE(random_state=30) 
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("[7] New Class Distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

#Model training
model = LogisticRegression(max_iter=1000, random_state=25)
model.fit(X_train_res, y_train_res)

#Prediction
y_pred = model.predict(X_test)

# [10] Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

import joblib
joblib.dump(model, 'logistic_regression_fraud_model.pkl')

print("[10] Results:")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print("Confusion Matrix:\n", cm)
