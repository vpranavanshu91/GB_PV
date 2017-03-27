#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 23:25:17 2017

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

"""
Data prep for sentence level NN processing
"""

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(sentences, padding_word="<PAD/>",length = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if length is None:
        sequence_length = max(len(x.split()) for x in sentences)
    else:
        sequence_length = length
    print('PAD size : ',sequence_length )
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence.split())
        new_sentence = sentence + ' ' + ' '.join([padding_word] * num_padding)
        padded_sentences.append(new_sentence.split())
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
#    sentences = [sen.split() for sen in sentences ]
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    # Switching out pad to index 0
    pad_ind = vocabulary_inv.index('<PAD/>')
    first = vocabulary_inv[0]
    vocabulary_inv[pad_ind] = first
    vocabulary_inv[0] = '<PAD/>'

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    print('Index of PAD : ', vocabulary.get('<PAD/>'))

    return [vocabulary, vocabulary_inv]

def build_input_data(q1, q2, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x1 = np.array([[vocabulary.get(word,0) for word in sentence] for sentence in q1])
    x2 = np.array([[vocabulary.get(word,0) for word in sentence] for sentence in q2])
    y = np.array(labels)
    return [x1, x2, y]

#%%
train =  pd.read_csv('train-2.csv')
holdout = train.sample(0.30* len(train))
holdout.to_csv('GB_PV/001_data/holdout.csv')
train = train[~train.index.isin(holdout.index)]
train.to_csv('GB_PV/001_data/train.csv')
#%%

q1 = Parallel(n_jobs=4)(delayed(clean_str)(str(q)) for q in train.question1)
q2 = Parallel(n_jobs=4)(delayed(clean_str)(str(q)) for q in train.question2)
all_questions = np.vstack([q1,q2]).ravel()
all_questions = pad_sentences(all_questions)# this takes a while!!
half_ind = len(all_questions)/2
q1 = all_questions[0:half_ind]
q2 = all_questions[half_ind:]
vocabulary, vocabulary_inv = build_vocab(all_questions)
x1,x2,y = build_input_data(q1,q2,train.is_duplicate,vocabulary)
#%%

pickle.dump((vocabulary,vocabulary_inv),open('GB_PV/001_data/001_vocabs.pickle','wb'))
pickle.dump((x1,x2,y),open('GB_PV/001_data/001_indexed_train_data.pickle','wb'))
