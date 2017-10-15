#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
from math import log


class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth):
        self.smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):
        for label in range(0, max(y) + 1):
            self._Nfeat.append([])   #create array for feature vectors to be grouped by labels
        
        for labels in range(0, len(X)):
            self._Nfeat[y[labels]].append(X[labels]) # group feature vectors by label
        
        for labelArray in self._Nfeat: 
            classProbability = float(len(labelArray) + self.smooth) / float(len(X) + (self.smooth)*2)
            self._class_prob.append(classProbability)  #append probabilities of each class to _class_prob array
        
        for feature in self._Nfeat:
            probOfFeatureGivenClass = []
            for i in range(0, len(feature[0])):
                    n = 0
                    for j in range(0, len(feature)):
                        if(feature[j][i] > 0):
                            n = n + 1
                    probOfFeatureGivenClass.append(float(n + self.smooth)/ float(len(feature) + (self.smooth*2)))
            self._feat_prob.append(probOfFeatureGivenClass)
        return   

    def predict(self, X):
        list = [] * len(X) 

        for feature in X:
            predicted_prob = -float('inf')
            best_predicted_prob = None
            for i in range(0, len(self._feat_prob)):
                res = log(self._class_prob[i])
                for j in range(0, len(feature)):
                    if feature[j] > 0:
                        res += log(self._feat_prob[i][j])
                    else:
                        res += log(1 - self._feat_prob[i][j])
                    if predicted_prob < res:
                        predicted_prob = res
                        best_predicted_prob = i
            list.append(best_predicted_prob)

        return list

class MyMultinomialBayesClassifier():
    # For graduate students only
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        # Your code goes here.
        return

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        return np.zeros([X.shape[0],1])
        


""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=False)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
print 'alpha=%i accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0))

