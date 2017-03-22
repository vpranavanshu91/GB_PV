#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 23:38:09 2017

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
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,Flatten, Reshape,Input
from keras.layers import Embedding,Convolution1D,MaxPooling1D,Convolution2D,MaxPooling2D,Merge,LSTM
from keras.optimizers import Adagrad

#%%
vocabulary,vocabulary_inv = pickle.load(open('GB_PV/001_data/001_vocabs.pickle','rb'))
x1,x2,y = pickle.load(open('GB_PV/001_data/001_indexed_train_data.pickle','rb'))
#%%

# The amazon embeddings
embd = pickle.load(open('/Volumes/Data/Email_Sentiment_Analysis/SA/Data/amazon_embd.pck','rb'))
embedding_weights = [np.array([embd[w] if w in embd \
                                       else np.random.uniform(-0.25, 0.25, 300) \
                                   for w in vocabulary_inv])]
                                   
                                   
words_inEmbedding = [w for w in vocabulary_inv if w in embd]
len(words_inEmbedding)
#%%

embedding_size = 300
vocab_size = len(vocabulary)
nb_feature_maps = [100,50,25]
maxlen = 1
nb_classes = 2
n_gram = 2

filter_sizes = [3,2,2]

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=maxlen,weights=embedding_weights, trainable = False))

model.add(Convolution1D(nb_filter = nb_feature_maps[0],  filter_length = filter_sizes[0], border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=1))

model.add(Convolution1D(nb_filter = nb_feature_maps[1],  filter_length = filter_sizes[1], border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=1))

#model.add(Convolution1D(nb_filter = nb_feature_maps[2],  filter_length = filter_sizes[2], border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_length=1))

#model.add(LSTM(78))

model.add(Dropout(0.25))
#model.add(Flatten())
model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adagrad' ,metrics=['accuracy'])

#%%
batch_size = 78
num_epochs = 4

model.fit(train_sens_w, train_labels_w, batch_size=batch_size,
              nb_epoch=num_epochs, validation_data=(test_sens_w,test_labels_w),shuffle=False)

#%%
def save_Keras_model(model,path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+'.h5')
    print("Saved model to disk")

save_Keras_model(model,'GB_PV/001_data/001_basic_CNN_amazon.model')