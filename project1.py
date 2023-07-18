#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Import dataset

# In[2]:


df = pd.read_csv('tested.csv')
df


# # Data Analysing

# In[3]:


df.info()


# In[4]:


df.isnull()


# In[5]:


#null values find
df.isnull().sum()


# In[6]:


#check Outliers
sns.heatmap(df.isnull(), yticklabels=False, annot=True)


# In[7]:


#find columns name
print(df.columns)


# In[8]:


print(df.dtypes)


# In[9]:


print(df.describe())


# In[10]:


print(df.head(10))


# In[11]:


df.count()


# In[12]:


# Cleaning the dataset for build my model
df = df.dropna()
df


# In[13]:


# After cleaning my columns names
print(df.columns)


# In[14]:


#After cleaning check Outliers 
sns.heatmap(df.isnull(), yticklabels=False, annot=True)


# # Data Visualization  

# In[15]:


df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[16]:


# For example, to create a scatter plot of 'Price' vs 'Kilometer'
plt.scatter(df['PassengerId'], df['Sex'])
plt.xlabel('PassengerId')
plt.ylabel('Sex')
plt.title('PassengerId vs Sex')
plt.show()


# In[17]:


# For example, to create a scatter plot of 'Price' vs 'Kilometer'
plt.scatter(df['Ticket'], df['Fare'])
plt.xlabel('Ticket')
plt.ylabel('Fare')
plt.title('Ticket vs Fare')
plt.show()


# In[18]:


# Step 2: Group the data by 'Survived' and count the number of occurrences
survival_counts = df['Survived'].value_counts()

# Step 3: Create the bar chart
labels = ['Not Survived', 'Survived']
values = survival_counts.values

plt.bar(labels, values)
plt.xlabel('Survival')
plt.ylabel('Count')
plt.title('Survival Count in Titanic Dataset')
plt.show()


# In[19]:


# Step 2: Group the data by 'Sex' and 'Age' and count the number of occurrences
sex_age_counts = df.groupby(['Sex', 'Age']).size().reset_index(name='Count')

# Step 3: Create the bar chart
labels = sex_age_counts['Sex'].astype(str) + ', ' + sex_age_counts['Age'].astype(int).astype(str)
values = sex_age_counts['Count']

plt.bar(labels, values)
plt.xlabel('Sex, Age')
plt.ylabel('Count')
plt.title('Count of Passengers by Sex and Age')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# # Logestic Regression Model

# In[23]:


# Step 2: Explore and preprocess the data
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Convert categorical features to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Step 3: Split the data into training and testing sets
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[24]:


df = model.predict(X_test)
print(df)


# In[25]:


df = model.predict(X_train)
print(df)


# In[26]:


print(X_train.shape, y_train.shape, X_test.shape)


# 

# In[ ]:





# In[ ]:





# In[ ]:




