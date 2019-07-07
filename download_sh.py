#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:51:24 2019

@author: rishabh
"""

import requests
import json
import time
import pandas as pd

# changing the user-agent 
headers = {'User-agent': 'E Bot 1.0'}

# setting after = None so I can start from the first page
after = None

# initializing an empty list of posts
shower_thoughts = []

# assigning the url
url = "https://www.reddit.com/r/Showerthoughts/.json"

# in order to get around 500 posts, run the loop 20 times
for i in range(20):
    
    # print index so I can check that it's proceeding
    print(i)
    
    # parameters are null for the first page
    if after == None:
        params = {}    
    else:
        # otherwise I have to change the parameter 'after' so it does not start all over again
        params = {'after': after}
        
    # generating the request  
    res = requests.get(url, params = params, headers = headers)
    
    # if there are no errors
    if res.status_code == 200:
        
        # get the json file
        sh_thoughts_json = res.json()
        
        # extend the list of posts with the children dictionary
        shower_thoughts.extend(sh_thoughts_json['data']['children'])
        
        # reassign the after parameter so the loop can move to the next set of posts
        after = sh_thoughts_json['data']['after']
    
    # if there is an error, print the type and break the loop
    else:
        print(res.status_code)
        break
        
    # 2 seconds break before pinging the server again
    time.sleep(2)
    
    
# the range of the iteration starts from 2 because the first 2 
# elements are not posts but instructions related to the page
shower_thoughts_list = [shower_thoughts[i]['data']['title'] + ' ' +
                        shower_thoughts[i]['data']['selftext'] 
                        for i in range(2,len(shower_thoughts))]

# creating the dataframe from the list
st_df = pd.DataFrame(shower_thoughts_list, columns=['post'])

# adding the subreddit column
st_df['subreddit'] = 'shower_thoughts'

# checking
st_df.head()

# changing the user-agent 
headers = {'User-agent': 'E Bot 1.0'}

# setting after = None so I can start from the first page
after = None

# initializing an empty list of posts
deep_philosophy = []

# assigning the url
url = "https://www.reddit.com/r/DeepPhilosophy/.json"

# in order to get around 500 posts, run the loop 20 times
for i in range(20):
    
    # print index so I can check that it's proceeding
    print(i)
    
    # parameters are null for the first page
    if after == None:
        params = {}    
    else:
        # otherwise I have to change the parameter 'after' so it does not start all over again
        params = {'after': after}
        
    # generating the request  
    res = requests.get(url, params = params, headers = headers)
    
    # if there are no errors
    if res.status_code == 200:
        
        # get the json file
        deep_philosophy_json = res.json()
        
        # extend the list of posts with the children dictionary
        deep_philosophy.extend(deep_philosophy_json['data']['children'])
        
        # reassign the after parameter so the loop can move to the next set of posts
        after = deep_philosophy_json['data']['after']
    
    # if there is an error, print the type and break the loop
    else:
        print(res.status_code)
        break
        
    # 2 seconds break before pinging the server again
    time.sleep(2)
    # the range of the iteration starts from 2 because the first 2 
# elements are not posts but instructions related to the page
deep_philosophy_list = [deep_philosophy[i]['data']['title'] + ' ' +
                        deep_philosophy[i]['data']['selftext'] 
                        for i in range(2,len(deep_philosophy))]

# creating the dataframe from the list
dp_df = pd.DataFrame(deep_philosophy_list, columns=['post'])

# adding the subreddit column
dp_df['subreddit'] = 'deep_philosophy'

# checking
dp_df.head()


df = st_df.append(dp_df, ignore_index=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# initializing the stop-words list
my_stopwords = stopwords.words('english')

# adding words that are in the vectorizer
# but are not actual words (unicode characters)
my_stopwords.extend(['amp','x200b','\n'])

# replacing the target values with 1s and 0s
# 1 for 'shower_thoughts' and 0 for 'deep_philosophy'
df['subreddit'].replace({'shower_thoughts': 1, 'deep_philosophy': 0}, inplace = True)

# defining X and y
X = df['post']
y = df['subreddit']

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2)

# instatiating the TfidfVectorizer
tfidvec = TfidfVectorizer(stop_words = my_stopwords)

# getting the sparse matrices for X_train and X_test
tfidvec.fit(X_train)
X_train_tfidf = tfidvec.transform(X_train)
X_test_tfidf = tfidvec.transform(X_test)

from sklearn.metrics import confusion_matrix

def conf_matrix(model, X_test):
    y_pred = model.predict(X_test)            # calculate predictions
    cm = confusion_matrix(y_test, y_pred)     # defining the confusion matrix
    tn, fp, fn, tp = cm.ravel()               # assigning the elements of the confusion matrix to variables
    print(f"True Negatives: {tn}")            # print those variables
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")            # return the confusion matrix as dataframe
    return pd.DataFrame(cm, 
                        columns = ['Pred Deep Philosophy','Pred Shower Thoughts'], 
                        index = ['Act Deep Philosphy', 'Act Shower Thoughts'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


steps = [
    ("vectorizer", TfidfVectorizer(stop_words=my_stopwords)),
    ("rf", RandomForestClassifier())
]

# instatiating the pipeline
pipe = Pipeline(steps)

# setting the values for the RandomForest model
grid_params = {
    "vectorizer__max_features": [2000, 3000, 4000],
    "vectorizer__ngram_range":[(1,1), (1,2)],
    "rf__n_estimators": [2500, 3000, 3500],
    "rf__max_depth": [17, 18, 19, 20],
    "rf__min_samples_leaf": [1, 2, 3]
}
# grid search
gs = GridSearchCV(pipe, grid_params, verbose=1, n_jobs=2)
results = gs.fit(X_train, y_train)


