#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
import string
import re
import math
import statistics
from math import sqrt
import numpy as np
import random
import time
import pandas as pd
import nltk, re, pprint
#nltk.download('punkt')
from nltk import word_tokenize
random.seed(1)
np.random.seed(1)
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import pyplot as plt
import collections
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import pickle

os.chdir('C:/Users/torna/Documents/StatApp/StatApp')
#os.chdir('/Users/alainquartierlatente/Desktop/Ensae/StatApp')
#os.chdir('/home/aqlt/Documents/Ensae/StatApp')
#os.chdir('C:/Users/Kim Antunez/Documents/Projets_autres/StatApp')


# # Sur 100K

# In[3]:


nom_dossier = "100k"
# Penser à changer selon taille
with open("data/corpus_trie%s.file" %nom_dossier, "rb") as f:
    corpus = pickle.load(f) 
ens_tweets = [phrase.split() for phrase in corpus]
phrases = ens_tweets
print(len(phrases))


# In[6]:


phrases[0:3]


# In[63]:


from gensim.models import word2vec
model = word2vec.Word2Vec(phrases, size=20, window=3,negative=5,alpha=0.01,seed=1,
                          min_count=5, workers=4, iter=10)
# phrases = les phrases du corpus
# size = dimension du vecteur
# window = fenetre
# negative = nb de neg samples
# alpha = learning rate
# seed 
# min_count = fréquence min des mots qui nous intéresse
# workers = nb de cores sur l'ordi
# iter = nb epoch
model.corpus_count


# In[52]:


vocab = model.wv.vocab
list(vocab)[:5]


# In[53]:


model.wv['wesh']


# In[54]:


model.similarity("homme","femme")


# In[65]:


def cos_distance(u, v):
    return (np.dot(u, v)  / (math.sqrt(np.dot(u, u)) *  (math.sqrt(np.dot(v, v)))))

def eucl_distance(u, v):
    return (np.linalg.norm(u-v))


# In[66]:


def distance_mots(word1,word2):
    if word1 in vocab and word2 in vocab:
        word_distance = model.similarity(word1,word2)
    else:
         word_distance = float('nan')
    return word_distance
distance_mots_v = np.vectorize(distance_mots)


# In[57]:


distance_mots("homme","femme")


# In[58]:


df_base = pd.read_csv('data_bis/word_similarity.csv', sep=";")
df_base


# In[59]:


distance_mots_v = np.vectorize(distance_mots)
df = df_base
df["corr_word2vec_cos"] = distance_mots_v(df["word1"],df["word2"])
print(len(df))
df = df.dropna()
print(len(df))
df


# In[60]:


from scipy.stats import spearmanr
#On fait des tests à 5 % pour la distance cosinus
alpha = 0.05
corr, p_value = spearmanr(df["corr"], df["corr_word2vec_cos"])
if p_value > alpha:
    print('Le résultat de word2vec COSINUS est différent de celui du human judgement (non rejet de H0 = non corrélation) p=%.3f' % p_value,'/ Valeur de la corrélation : %.3f'% corr)
else:
    print('Le résultat de word2vec COSINUS est semblable celui du human judgement (rejet de H0 = non corrélation) p=%.3f' % p_value,'/ Valeur de la corrélation : %.3f'% corr)

print("\n")


# # Sur l'ensemble

# In[67]:


nom_dossier = "ens"
# Penser à changer selon taille
with open("data/corpus_trie%s.file" %nom_dossier, "rb") as f:
    corpus = pickle.load(f) 
ens_tweets = [phrase.split() for phrase in corpus]
phrases = ens_tweets
print(len(phrases))


# In[72]:


model = word2vec.Word2Vec(phrases, size=100, window=3,negative=5,alpha=0.01,seed=1,
                          min_count=5, workers=4, iter=100)
# phrases = les phrases du corpus
# size = dimension du vecteur
# window = fenetre
# negative = nb de neg samples
# alpha = learning rate
# seed 
# min_count = fréquence min des mots qui nous intéresse
# workers = nb de cores sur l'ordi
# iter = nb epoch
model.corpus_count


# In[73]:


vocab = model.wv.vocab
list(vocab)[:5]


# In[74]:


df_base = pd.read_csv('data_bis/word_similarity.csv', sep=";")
distance_mots_v = np.vectorize(distance_mots)
df = df_base
df["corr_word2vec_cos"] = distance_mots_v(df["word1"],df["word2"])
print(len(df))
df = df.dropna()
print(len(df))
df


# In[75]:


from scipy.stats import spearmanr
#On fait des tests à 5 % pour la distance cosinus
alpha = 0.05
corr, p_value = spearmanr(df["corr"], df["corr_word2vec_cos"])
if p_value > alpha:
    print('Le résultat de word2vec COSINUS est différent de celui du human judgement (non rejet de H0 (= non corrélation) p=%.3f' % p_value,'/ Valeur de la corrélation : %.3f'% corr)
else:
    print('Le résultat de word2vec COSINUS est semblable celui du human judgement (rejet de H0 (= non corrélation)) p=%.3f' % p_value,'/ Valeur de la corrélation : %.3f'% corr)

print("\n")

