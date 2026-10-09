#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import joblib


# In[5]:


df = pd.read_csv(r"C:\Users\HP\Downloads\creditcard.csv\creditcard.csv")


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


print(df['Class'].value_counts())


# In[11]:


plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Genuine Transactions")
plt.show()


# In[12]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[13]:


scaler = StandardScaler()

df['Amount'] = scaler.fit_transform(df[['Amount']])


# In[14]:


X = df.drop('Class', axis=1)

y = df['Class']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[16]:


lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

lr.fit(X_train, y_train)


# In[18]:


lr_pred = lr.predict(X_test)


# In[19]:


print("Accuracy:", accuracy_score(y_test, lr_pred))


# In[20]:


print(classification_report(y_test, lr_pred))


# In[21]:


cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5,4))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[22]:


lr_prob = lr.predict_proba(X_test)[:,1]

print("ROC AUC:", roc_auc_score(y_test, lr_prob))


# In[ ]:


rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)


# In[ ]:


rf_pred = rf.predict(X_test)


# In[26]:


rf = RandomForestClassifier(
    n_estimators=30,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# In[27]:


rf_pred = rf.predict(X_test)


# In[28]:


print("Accuracy:", accuracy_score(y_test, rf_pred))


# In[29]:


print(classification_report(y_test, rf_pred))


# In[30]:


cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Greens')

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[31]:


rf_prob = rf.predict_proba(X_test)[:,1]

print("ROC AUC:", roc_auc_score(y_test, rf_prob))


# In[32]:


fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()


# In[33]:


importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))

importance.head(15).plot(kind='bar')

plt.title("Top 15 Important Features")

plt.show()


# In[34]:


joblib.dump(rf, "fraud_detection_model.pkl")

print("Model Saved Successfully")


# In[35]:


model = joblib.load("fraud_detection_model.pkl")


# In[36]:


sample = X_test.iloc[0:1]

prediction = model.predict(sample)

if prediction[0] == 0:
    print("Genuine Transaction")
else:
    print("Fraudulent Transaction")


# In[37]:


results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "ROC AUC": [
        roc_auc_score(y_test, lr_prob),
        roc_auc_score(y_test, rf_prob)
    ]
})

print(results)


# In[ ]:




