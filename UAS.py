#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[7]:


DataSet = pd.DataFrame()
DataSet = pd.read_csv(r"C:/Users/Asus ROG/Downloads/UAS/diabetes.csv")
DataSet.head(10)


# In[8]:


for column in DataSet.columns[1:-3]:
    DataSet[column].replace(0,np.NaN, inplace=True)
    DataSet[column].fillna(round(DataSet[column].mean(skipna=True)), inplace=True)
DataSet.head(10)


# In[9]:


X = DataSet.drop("Outcome",axis=1)
y = DataSet["Outcome"]
# Split data into 20% for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[10]:


neighbor = KNeighborsClassifier(n_neighbors=5,metric="euclidean")
neighbor.fit(X_train,y_train)


# In[11]:


y_pred = neighbor.predict(X_test)
y_pred


# In[12]:


print(confusion_matrix(y_pred,y_test))


# In[13]:


print(accuracy_score(y_pred,y_test))


# In[14]:


print(classification_report(y_pred,y_test))

