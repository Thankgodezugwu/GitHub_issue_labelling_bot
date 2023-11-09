#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[6]:


get_ipython().system('wget https://tickettagger.blob.core.windows.net/datasets/dataset-labels-top3-30k-real.txt')


# In[7]:


get_ipython().system('pip install requests')


# In[10]:


#download the dataset from ticket tagger
import requests

url = "https://tickettagger.blob.core.windows.net/datasets/dataset-labels-top3-30k-real.txt"
response = requests.get(url)

with open("git_issue.csv", "wb") as file:
    file.write(response.content)


# In[13]:


#import dataset
df = pd.read_csv("git_issue.csv", header = None)


# In[14]:


df.head()


# In[17]:


#Text cleaning and extractrion
#extraxt and split the labels using regular expression
#split into enhancement,bug and question
df[0].str.split(r'(__label__enhancement)|(__label__bug)|(__label__question)', expand = True)


# In[18]:


df_new = df[0].str.split(r'(__label__enhancement)|(__label__bug)|(__label__question)', expand = True)


# ## Feature Engineering|extraction
# 

# In[20]:


#Extract only rows with enhancement
df_new[df_new[1] == '__label__enhancement'][[1,4]]


# In[30]:


enh_df = df_new[df_new[1] == '__label__enhancement'][[1,4]]


# In[31]:


#Extract only rows with bug
df_new[df_new[2] == '__label__bug'][[2,4]]


# In[32]:


bug_df = df_new[df_new[2] == '__label__bug'][[2,4]]


# In[33]:


#Extract only rows with question
df_new[df_new[3] == '__label__question'][[3,4]]


# In[36]:


question_df = df_new[df_new[3] == '__label__question'][[3,4]]


# ### Assign names to each of the columns and merge them together 

# In[37]:


#Assign names to each columns
enh_df.columns = ['label', 'description']
bug_df.columns = ['label', 'description']
question_df.columns = ['label', 'description']


# In[42]:


#concatinate the columns
df = pd.concat([enh_df, bug_df, question_df])


# In[43]:


df.head()


# In[44]:


df.shape


# In[45]:


#save the dataset 
df.to_csv('github-issue.csv')


# ### Clean the data

# In[47]:


#remove the __label__ 
df['label'].str.replace('__label__','')


# In[48]:


df['label'] = df['label'].str.replace('__label__','')


# In[49]:


#save the dataset 
df.to_csv('github-issue-label.csv')


# ### Class Distribution Analysis

# In[52]:


df['label'].value_counts()


# In[56]:


fig = plt.figure(figsize = (10,5))
sns.countplot(data = df, x = 'label')
plt.title('Distribution of the labels')
plt.show()


# ### Text cleaning 

# In[62]:


df.iloc[0].description


# In[63]:


get_ipython().system('pip install neattext')


# In[65]:


#remove stop words and covert everything to lower case
import neattext.functions as nfx
df['description'] = df['description'].apply(lambda x: nfx.remove_stopwords(str(x).lower()))


# Because, some of the special characters are vital informations we may not consider removing it

# ### Build model using pipline

# In[100]:


#split the dataset
X = df['description']
y = df['label']


# In[101]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)


# In[102]:


#create pipeline models
pipe_base = Pipeline(steps = [('cv',CountVectorizer()), ('dummy', DummyClassifier())])
pipe_nb = Pipeline(steps = [('cv',CountVectorizer()), ('nb', MultinomialNB())])
pipe_dt = Pipeline(steps = [('cv',CountVectorizer()), ('dummy', DecisionTreeClassifier())])
pipe_lr = Pipeline(steps = [('cv',CountVectorizer()), ('dummy', LogisticRegression())])
pipe_rf = Pipeline(steps = [('cv',CountVectorizer()), ('dummy', RandomForestClassifier())])


# In[103]:


#build base model
pipe_base.fit(X_train, y_train)


# In[104]:


#accuracy
pipe_base.score(X_test,y_test)


# In[105]:


#build naive bayes model
pipe_nb.fit(X_train, y_train)


# In[106]:


#accuracy
pipe_nb.score(X_test,y_test)


# In[107]:


#build decision tree model
pipe_dt.fit(X_train, y_train)


# In[108]:


#accuracy
pipe_dt.score(X_test,y_test)


# In[109]:


#build logistic regression model
pipe_lr.fit(X_train, y_train)


# In[110]:


#accuracy
pipe_lr.score(X_test,y_test)


# In[84]:


#build random forest model
pipe_rf.fit(X_train, y_train)


# In[85]:


#accuracy
pipe_rf.score(X_test,y_test)
#because of excecution time, I will not use random forest


# In[111]:


y_pred = pipe_lr.predict(X_test)

print(classification_report(y_test, y_pred))


# In[112]:


#accuracy
pipe_lr.score(y_test,y_pred)


# In[118]:


fig = plt.figure(figsize = (5,5))
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.title('Confusion matrix display')
plt.show()


# In[119]:


#save the model
import joblib


# In[120]:


#save the model
model_file = open("pipe_lr_gh_issue_classifier.pkl","wb")
joblib.dump(pipe_lr,model_file)
model_file.close()


# In[121]:


#predict unseen data
#this example is actually a bug in the issue label
ex1 = "Input Widgets re-run the entire app every time"
pipe_lr.predict([ex1])


# In[123]:


#example 2 is enhancement
exp2 = "Request: on session start/shutdown hooks"
pipe_lr.predict([exp2])


# In[ ]:




