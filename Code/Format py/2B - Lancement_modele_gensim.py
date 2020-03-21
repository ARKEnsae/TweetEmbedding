#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
import datetime

import gensim
from gensim.models import word2vec

#os.chdir('C:/Users/torna/Documents/StatApp/StatApp')
#os.chdir('/Users/alainquartierlatente/Desktop/Ensae/StatApp')
#os.chdir('/home/aqlt/Documents/Ensae/StatApp')
os.chdir('C:/Users/Kim Antunez/Documents/Projets_autres/StatApp')

#nom_dossier = "ens"
nom_dossier = "100k"


# In[7]:


# Penser à changer selon taille
with open("data/corpus_trie%s.file" %nom_dossier, "rb") as f:
    corpus = pickle.load(f) 
ens_tweets = [phrase.split() for phrase in corpus]
phrases = ens_tweets
print(len(phrases))


# # Lancement du modèle
# 
# ### Nouvelle simulation 
# 
# Paramètres fixés : le nombre de mots tirés dans le *negative sampling* et la proba utilisé

# Paramètres à modifier :

# In[8]:


seeds = [1, 5, 10, 15, 20, 25]

# renseigner dans l'ordre : dim, epoch, window, learning_rate
simuls = [(20,10,3,0.01),(50,10,3,0.01)]

# Rq : numero_simulation n'est plus utilisé désormais ! 


# In[9]:


for dim, epoch, window, learning_rate in simuls: #suppr

    for seed in seeds:

        start = time.time()

        model = word2vec.Word2Vec(phrases, size=dim, window=window,negative=5,
                                  alpha=learning_rate,min_alpha=learning_rate, seed=seed,
                          min_count=0, workers=4, iter=epoch)
        # Kim : j'ai mis min_count = 0, devons-nous mettre 4 ?
        
        end = time.time()
        # Affichage temps de tournage du modèle
        currentDT = datetime.datetime.now()
        print("seed {} à {}:{} : {}".format(seed, currentDT.hour, currentDT.minute, round((end - start)/60, 2)))

        # Sauvegarde 
        chemin = "data/{}/gensim/dim{}_ep{}_w{}_lr{}_seed{}".format(nom_dossier, dim, epoch, window, str(learning_rate)[2:], seed)
        if not os.path.isdir(chemin):
            os.mkdir(chemin)

        model.save(chemin + "/word2vec.model")


# phrases = les phrases du corpus
# size = dimension du vecteur
# window = fenetre
# negative = nb de neg samples
# alpha = learning rate
# seed 
# min_count = fréquence min des mots qui nous intéresse
# workers = nb de cores sur l'ordi
# iter = nb epoch


# ### Simulation déjà effectuée (faire de nouvelles epoch)
# 
# CETTE PARTIE BUG ET EST A RECODER !
# 
# Paramètres à modifier :

# In[5]:


epoch = 1 #Nouveau nombre d'epoch à effectuer
learning_rate = 0.01
numero_simulation = 100
seed = 1
version = 1 #version du fichier à modifier : pex version = "1" pour word2vec1.model


# In[ ]:


chemin = "data/" + nom_dossier + "/gensim/Simulation_" + str(numero_simulation) + "_seed" + str(seed)       
model = gensim.models.keyedvectors.KeyedVectors.load(chemin + "/word2vec" + str(version) +".model")  

start = time.time()

#### NE MARCHE PAS ! 
#model.train(phrases, epochs=epoch, total_examples=model.corpus_count, 
#            start_alpha=learning_rate, end_alpha=learning_rate) #, compute_loss=True
   
end = time.time()

# Sauvegarde 
#model.save(chemin + "/word2vec" + str(version+1) +".model")


# #  Quelques fonctions intégrées dans gensim (non utilisées)

# In[9]:


model.corpus_count


# In[10]:


vocab = model.wv.vocab
list(vocab)[:5]


# In[11]:


model.wv['wesh']


# In[12]:


model.similarity("homme","femme")


# Biblio : 
#     
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# http://kavita-ganesan.com/how-to-incorporate-phrases-into-word2vec-a-text-mining-approach/#.XmvT7XIiE2x
# https://rare-technologies.com/word2vec-tutorial/
# https://radimrehurek.com/gensim/models/word2vec.html
