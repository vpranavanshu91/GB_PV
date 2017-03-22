#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 21:28:52 2017

@author: gautamborgohain
"""
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re,pickle
from joblib import Parallel,delayed
import json
import scipy.stats as stats
import itertools
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation as cv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score,log_loss
from sklearn.metrics import roc_curve, auc
#%%
train =  pd.read_csv('train-2.csv')
test =  pd.read_csv('test.csv')
print('Train size{}'.format(len(train)))
print('Test size{}'.format(len(test)))
del test
#%%
"""
Basic fuzzy matching of the questions

"""
from Levenshtein import *

def compute_lev_distance(sen1,sen2):
    return ratio(sen1.lower(),sen2.lower())
    
train_lev_scores = Parallel(n_jobs=4)(delayed(compute_lev_distance)(str(sen1),str(sen2)) for sen1,sen2 in zip(train.question1,train.question2))
train_lev_preds = [1 if score>0.6 else 0 for score in train_lev_scores]

print('Accuracy : {}'.format(accuracy_score(train_lev_preds,train.is_duplicate)))
print('LogLoss : {}'.format(log_loss(train.is_duplicate,train_lev_scores)))




