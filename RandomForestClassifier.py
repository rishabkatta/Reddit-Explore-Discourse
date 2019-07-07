#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:51:24 2019

@author: rishabh
"""

import requests
import time
import pandas as pd

headers = {'User-agent': 'E Bot 1.0'}  # declaring the bot that downloads our posts
after = None #after parameter to get next 25 posts every time
unpopular_opinion = []
url = "https://www.reddit.com/r/unpopularopinion/.json"

# get 500 posts by getting 25 posts everytime in a loop
for i in range(20):
    print(i)
    if after == None:  # after is none for the first page
        params = {}    
    else:
        params = {'after': after} #if not first page, change after
    res = requests.get(url, params = params, headers = headers) # sending the request to reddit to get posts
    
    if res.status_code == 200: #200 code means request got processed successfully
        unpopular_opinion_json = res.json() # get the posts in json format
        unpopular_opinion.extend(unpopular_opinion_json['data']['children']) #add posts to the list
        after = unpopular_opinion_json['data']['after'] #change after to the next set of posts.
    else:
        print(res.status_code)
        break
    time.sleep(2) #sleep 2 seconds before pinging the server again.
    
unpopular_opinion_list = [unpopular_opinion[i]['data']['title'] + ' ' +
                        unpopular_opinion[i]['data']['selftext'] 
                        for i in range(2,len(unpopular_opinion))]

uo_df = pd.DataFrame(unpopular_opinion_list, columns=['post'])      #creating a dataframe from the list.
uo_df['subreddit'] = 'unpopular_opinion'
uo_df.head()


headers = {'User-agent': 'E Bot 1.0'}  # changing the user-agent 
after = None #after parameter to get next 25 posts every time
world_news = []
url = "https://www.reddit.com/r/worldnews/.json"

# get 500 posts by getting 25 posts everytime in a loop
for i in range(20):
    print(i)
    if after == None:  # after is none for the first page
        params = {}    
    else:
        params = {'after': after} #if not first page, change after
    res = requests.get(url, params = params, headers = headers) # sending the request to reddit to get posts
    
    if res.status_code == 200: #200 code means request got processed successfully
        world_news_json = res.json() # get the posts in json format
        world_news.extend(world_news_json['data']['children']) #add posts to the list
        after = world_news_json['data']['after'] #change after to the next set of posts.
    else:
        print(res.status_code)
        break
    time.sleep(2) #sleep 2 seconds before pinging the server again.
    
world_news_list = [world_news[i]['data']['title'] + ' ' +
                        world_news[i]['data']['selftext'] 
                        for i in range(2,len(world_news))]

wn_df = pd.DataFrame(world_news_list, columns=['post'])      #creating a dataframe from the list.
wn_df['subreddit'] = 'world_news'
wn_df.head()

df = uo_df.append(wn_df, ignore_index=True) #appending two dataframes and create a new dataframe

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

my_stopwords = stopwords.words('english')   # creating a stopwords list using nltk
my_stopwords.extend(['amp','x200b','\n']) #adding any additional unicode charecters that the posts text might contain
df['subreddit'].replace({'unpopular_opinion': 1, 'world_news': 0}, inplace = True) #replacing subreddit names with 1 or 0.

X = df['post']     #defining features
y = df['subreddit'] #defining labels
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2) #splitting training-test with 75-25
tfidvec = TfidfVectorizer(stop_words = my_stopwords) #creating vectors with word counts and "importance" values
tfidvec.fit(X_train)
X_train_tfidf = tfidvec.transform(X_train)
X_test_tfidf = tfidvec.transform(X_test)

#creating a confusion matrix to help visualize the performance of the algorithm.
from sklearn.metrics import confusion_matrix
def conf_matrix(model, X_test):
    y_pred = model.predict(X_test)            
    cm = confusion_matrix(y_test, y_pred)     
    tn, fp, fn, tp = cm.ravel()               
    print(f"True Negatives: {tn}")            
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")            
    return pd.DataFrame(cm, 
                        columns = ['Pred Unpopular Opinion','Pred World news'], 
                        index = ['Act Unpopular Opinion', 'Act World news'])

#Defining a Random Forest classifier and fitting & predicting the data using that classifier.
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train_tfidf,y_train)
y_pred=clf.predict(X_test_tfidf)
y_score=clf.predict_proba(X_test_tfidf)[:,1]
print(conf_matrix(clf,X_test_tfidf))


#printing Accuracy and Precision Recall score
from sklearn import metrics
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  


#plotting Precision Recall curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
precision, recall, _ = precision_recall_curve(y_test, y_score)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))


