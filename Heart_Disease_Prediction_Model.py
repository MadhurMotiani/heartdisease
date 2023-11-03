#!/usr/bin/env python
# coding: utf-8

# # Classification of a specific heart disease using machine learning techniques.

# #### Build a machine learning model(s), that can detect between a subject afflicted with heart disease and someone who is normal. Problems such as this are common in the healthcare field where such medical diagnoses can be made with the aid of machine learning and AI techniques, usually with much better accuracy. Hospitals and medical enterprises often employ specialists such as machine learning engineers and data scientists to carry out these tasks.

# #### Attribute Information: 
# 
# Using the 13 attributes which are already extracted, in the heart disease dataset, you are 
# expected to detect either the presence of or the absence of the heart disease in human 
# subjects. 
# There are 13 attributes: 
# 1.  age: age in years 
# 2. sex: sex (1 = male; 0 = female) 
# 3. cp: chest pain type 
# -- Value 0: typical angina 
# -- Value 1: atypical angina 
# -- Value 2: non-anginal pain 
# -- Value 3: asymptomatic 
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital) 
# 5. chol: serum cholesterol in mg/dl 
# 6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
# 7. restecg: resting electrocardiographic results 
# -- Value 0: normal 
# -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or 
# depression of > 0.05 mV) 
# -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' 
# criteria 
# 8. thalach: maximum heart rate achieved 
# 9. exang: exercise induced angina (1 = yes; 0 = no) 
# 10.  oldpeak = ST depression induced by exercise relative to rest 
# 11. slope: the slope of the peak exercise ST segment 
# -- Value 0: upsloping 
# -- Value 1: flat 
# -- Value 2: downsloping 
# 12. ca: number of major vessels (0-3) colored by flourosopy 
# 13.thal: 0 = normal; 1 = fixed defect; 2 = reversable defect 
# and the label 
# 14.  condition: 0 = no disease, 1 = disease

# ### Data Preprocessing:
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


data = pd.read_csv("data-problem-statement-1-heart-disease.csv")
df = data

df.head()

df.tail()

df.info()


print("Rows     : ", df.shape[0])
print("Columns  : ", df.shape[1])
print("\nFeatures : \n", df.columns.tolist())
print("\nMissing values :  ", df.isnull().sum())
print("\nUnique values :  \n", df.nunique())

categorical_val = []
continous_val = []
for column in df.columns:
    print("--------------------")
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

# Pairplot to visualize relationships between numerical variables
import matplotlib.pyplot as plt
sns.pairplot(df, hue='condition', diag_kind='kde')
plt.show()

import matplotlib
from matplotlib import pyplot as plt
plt.figure(figsize=(20,12))
sns.set_context('notebook',font_scale = 1)
sns.heatmap(df.corr(),annot=True,linewidth =2)
plt.tight_layout()

sns.countplot(data=df, x='condition')
plt.title('Heart Disease Presence (1) vs. Absence (0)')
plt.show()

data.describe()


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1)
sns.countplot(df['age'],hue=df['condition'])
plt.tight_layout()


# ### Understanding the Importance of each variable

# In[150]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

X = df.drop('condition', axis=1)
y = df['condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Importance scores
feature_importances = clf.feature_importances_

# Create a DataFrame to associate feature names with importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize the feature importance scores
print(feature_importance_df)

# Plot importance scores
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Scores')
plt.show()


# In[151]:


# Dropping the least important columns

newdf = df.drop(axis=1, columns= ['slope', 'exang'])
newdf


# ### Normalizing the Data

# In[152]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(newdf)

# Create a DataFrame with the normalized data
normalized_df = pd.DataFrame(data_normalized, columns=newdf.columns)
normalized_df.head()


# In[180]:


normalized_df.describe()


# In[153]:


# Select numeric columns for outlier detection
num_cols = normalized_df.columns

# Visualize outliers using box plots
plt.figure(figsize=(12, 6))
normalized_df[num_cols].boxplot(vert=False)
plt.title("Box Plot of Numeric Features")
plt.xlabel("Feature Value")
plt.show()


# ## MODEL SELECTION

# #### Logistic Regression

# In[178]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

X = normalized_df.drop("condition", axis=1)
y = normalized_df["condition"]

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the feature data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing and training a logistic regression model
print("LOGISTIC REGRESSION")
model = LogisticRegression()
model.fit(X_train, y_train)
# Making predictions on the test data
y_pred = model.predict(X_test)
# Evaluating the model
accuracy1 = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy1)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# #### K-Fold Cross Validation with SVM (K=10):

# In[167]:



import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

X = normalized_df.drop(columns=["condition"])
y = normalized_df["condition"]

# Defining the number of folds (k) for cross-validation
num_folds = 10

# Initializing a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initializing SVM model
svm_classifier = SVC(kernel="linear", C=1.0)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

# Lists to store accuracy scores for each fold
accuracy_scores = []

# Performing k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy and store it in the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_scores) / num_folds
print("K-Fold Crossvalidation  k=10 using SVM")
print(f"Accuracy: {average_accuracy}")
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# #### K-Fold Cross Validation with SVM (K=2):

# In[175]:


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

X = normalized_df.drop(columns=["condition"])
y = normalized_df["condition"]

# Defining the number of folds (k) for cross-validation
num_folds = 2

# Initializing a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initializing SVM model
svm_classifier = SVC(kernel="linear", C=1.0)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

# Lists to store accuracy scores for each fold
accuracy_scores = []

# Performing k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy and store it in the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_scores) / num_folds
print("K-Fold Crossvalidation k=02 using SVM")
print(f"Accuracy: {average_accuracy}")
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# ## ROC curve

# ##### The ROC curve is generated by plotting the TPR on the y-axis and the FPR on the x-axis for various threshold values. Each point on the curve represents the trade-off between sensitivity and specificity at a particular threshold. A diagonal line (the "no information" or random classifier line) is often shown on the ROC plot, and good classification models should be above this line.

# In[177]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Prepare the data for modeling
X = normalized_df.drop("condition", axis=1)
y = normalized_df["condition"]
X = pd.get_dummies(X, columns=["cp", "restecg", "thal"], drop_first=True)

# Split the data into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classification model (Random Forest in this example)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




