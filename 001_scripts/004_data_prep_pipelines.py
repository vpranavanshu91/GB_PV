#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:23:39 2017

@author: gautamborgohain
"""
import pandas as pd
import numpy as np
from Levenshtein import *
from joblib import Parallel,delayed

def compute_lev_distance(sen1,sen2):
    return ratio(sen1.lower(),sen2.lower())


import matplotlib.pyplot as plt
from sklearn import metrics

class plot_roc_binary():
    """
    Plot simple ROC
    call plot method
    """
    def plot(self,y,predicted):
        """
        
        :param y: True Labels
        :param predicted: Predicted Labels
        :return: plot
        """
        fpr, tpr, thresholds = metrics.roc_curve(y, predicted)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = {})'.format( metrics.auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
        return plt
import re
from sklearn.base import BaseEstimator, TransformerMixin

class clean_sent_pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self,options = None):
        if options is None :
            self.col1 = 'sentence_1'
            self.col2 = 'sentence_2'
        else:
            self.col1 = options.get('col1')
            self.col2 = options.get('col2')

#        
    
    def clean_str(self,string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = str(string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`;_:]", " ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    
    
    def transform(self,data, y=None):
        data[self.col1] = data[self.col1].map(lambda sen: self.clean_str(sen))
        data[self.col2] = data[self.col2].map(lambda sen: self.clean_str(sen))
        return data
    
    
    def fit(self, x, y=None):
        return self
    
    
    
import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

 
class count_vectorizer_pipeline(BaseEstimator, TransformerMixin):

    def __init__(self, options):
        if options is None :
            self.col = 'sentence'
        else:
            self.col = options.get('col')
        self.cv = CountVectorizer(lowercase = True,min_df =1,ngram_range = (1,2),stop_words = None,max_df = 1)

    def fit(self,X,y = None):
        sens = X[self.col]
        self.cv.fit(sens)
        return self

    def transform(self,X):
        sens = X[self.col]
        dmatrix = self.cv.transform(sens)
        return dmatrix

    def get_feature_names(self):
        return self.cv.get_feature_names()
    
class tfidf_vectorizer_pipeline(BaseEstimator, TransformerMixin):

    def __init__(self, options):
        if options is None :
            self.col = 'sentence'
        else:
            self.col = options.get('col')
        self.cv = TfidfVectorizer(lowercase = True,min_df =1,ngram_range = (1,2),stop_words = None,max_df = 1)

    def fit(self,X,y = None):
        sens = X[self.col]
        self.cv.fit(sens)
        return self

    def transform(self,X):
        sens = X[self.col]
        dmatrix = self.cv.transform(sens)
        return dmatrix

    def get_feature_names(self):
        return self.cv.get_feature_names()



from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, \
    roc_auc_score, auc, log_loss
from sklearn.cross_validation import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin
from Levenshtein import *

class fuzzy_feats_pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, options=None):
        if options is None :
            self.col1 = 'sentence_1'
            self.col2 = 'sentence_2'
        else:
            self.col1 = options.get('col1')
            self.col2 = options.get('col2')
    
    def fit(self,data,y = None):
        return self
    
    def transform(self,data,y = None):
        res = pd.DataFrame()
        sent_lev_ratio,set_ratio = [],[]
        for q1,q2 in zip(data[self.col1], data[self.col2]):
            sent_lev_ratio.append(ratio(q1,q2))
            set_ratio.append(setratio(q1.split(), q2.split()))
        res['sent_lev_ratio'] = sent_lev_ratio
        res['set_ratio'] = set_ratio
        return res.as_matrix()
    
    def get_feature_names(self):
        return ['sent_lev_ratio','set_ratio']
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import numpy as np

class sent_embd_feats_pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, options=None):
        if options is None :
            self.col = 'sentence'
        else:
            self.col = options.get('col')
        self.embd = pickle.load(open('/Volumes/Data/Email_Sentiment_Analysis/SA/Data/amazon_embd.pck','rb'))
    
    def sent2vec(self,s):
        words_embd = []
        for word in s.split():
            if word.isalpha() and word in self.embd:
                words_embd.append(self.embd[word])
                
        words_embd = [[float(n) for n in v] for v in words_embd]
        words_embd  = np.array(words_embd)    
        v = words_embd.sum(axis = 0)
        res = v / np.sqrt((v ** 2).sum())
        if type(res) == np.float64:
            res = np.random.uniform(-0.25, 0.25, 300)
        return res
    
    def fit(self,data,y = None):
        return self
    
    def transform(self,data,y = None):
        res = [self.sent2vec(s) for s in data[self.col]]
        return res
    
 
def compute_metrics(y_test,Preds):
    Acc = accuracy_score(y_true=y_test, y_pred=Preds)
    P = precision_score(y_true=y_test, y_pred=Preds)
    R = recall_score(y_true=y_test, y_pred=Preds)
    F = f1_score(y_true=y_test, y_pred=Preds)
    ll = log_loss(y_true=y_test, y_pred=Preds)
    print('Accuracy:,{}, Precision : {}, Recall: {}, F1Score: {}, Logloss: {}'.format(Acc, P, R, F, ll))
    print(classification_report(y_test, Preds))
    
    return Acc,P,R,F,ll
   
    
def run_pipeline(train,pipeline,cv):
    
    Accs,Ps,Rs,Fs,LLs = [],[],[],[],[]
    X = train[['question1','question2']]
    Y = train['is_duplicate']
    
    for i in range(cv):
        print('-' * 50)
        print('run ', i + 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        print('train and test shapes', X_train.shape, X_test.shape, np.array(y_train).shape, np.array(y_test).shape)
        X_train.index = range(len(X_train))
        y_train.index = range(len(y_train))
        X_test.index = range(len(X_test))
        y_test.index = range(len(y_test))
        
        print('Transforming train set....')
        feats_X = pipeline.fit_transform(X_train, y_train)
        print('Number of Features generated : ', feats_X.shape)
        
    
        
        print('Testing... ')
        Preds = pipeline.predict(X_test)
        
        Acc,P,R,F,ll = compute_metrics(y_test,Preds)
        Accs.append(Acc);
        Ps.append(P);
        Rs.append(R);
        Fs.append(F)
        LLs.append(ll)
        plot = plot_roc_binary().plot(y_test,Preds)
        train_lev_scores = Parallel(n_jobs=4)(delayed(compute_lev_distance)(str(sen1),str(sen2)) for sen1,sen2 in zip(X_test.question1,X_test.question2))
        train_lev_preds = [1 if score>0.6 else 0 for score in train_lev_scores]
        print('baseline:')
        Acc,P,R,F,ll = compute_metrics(y_test,train_lev_preds)
        plot1 = plot_roc_binary().plot(y_test,train_lev_preds)


