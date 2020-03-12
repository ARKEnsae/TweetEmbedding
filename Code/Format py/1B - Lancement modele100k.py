#!/usr/bin/env python
# coding: utf-8

# # Importation des données

# In[2]:


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

#os.chdir('C:/Users/torna/Documents/StatApp/StatApp')
#os.chdir('/Users/alainquartierlatente/Desktop/Ensae/StatApp')
#os.chdir('/home/aqlt/Documents/Ensae/StatApp')
os.chdir('C:/Users/Kim Antunez/Documents/Projets_autres/StatApp')
nom_dossier = "100k"
#nom_dossier = "ens" #ou


# On ne garde ici que les 100 000 premiers tweets

# In[3]:


# Penser à changer selon taille
with open("data/%s/vocabulary.file" %nom_dossier, "rb") as f:
    taille_vocab = len(pickle.load(f))


# In[4]:


# Fonction pour mettre à jour le graphique en direct
def live_plot(data, figsize=(7,5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.plot(data)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.show();


# # Lancement du modèle
# Paramètres fixés : la dimension, le nombre de mots tirés dans le *negative sampling* et la proba utilisé, l'affichage du graphique

# In[5]:


if nom_dossier == "ens":
    dim = 50
else:
    dim = 20
plot = True
epoch = 10


# Paramètres à modifier :

# In[8]:


#Kim
learning_rate = 0.01
window = 5
numero_simulation = 4
seeds = [1, 5, 10, 15, 20, 25]


# In[9]:


seed = 1
# On crée le dossier Simulation_{numero_simulation}
if not os.path.exists("data/%s/Simulation_%i_seed%i" %(nom_dossier, numero_simulation, seed)):
    os.mkdir("data/%s/Simulation_%i_seed%i" %(nom_dossier, numero_simulation, seed))
else:
    print("Attention : le dossier Simulation_%i_seed%i existe déjà" %(numero_simulation, seed))

torch.manual_seed(seed)
input = torch.randn(taille_vocab, dim)
output = torch.randn(taille_vocab, dim)
input = autograd.Variable(input, requires_grad=True)
output = autograd.Variable(output, requires_grad=True)

loss_tot = []
temps_par_epoch = []

start = time.time()
for i in range(epoch):
    loss_val = 0
    start_epoch = time.time()
    
    print("Simulation %i - Lecture du fichier data/%s/window_%i/epoch_%i.file" %(numero_simulation,nom_dossier, window, i))
    with open("data/%s/window_%i/epoch_%i.file" % (nom_dossier, window, i), "rb") as f:
        test_sample = pickle.load(f)

    for focus, context, neg_sample in test_sample:
        data = torch.matmul(input[focus,], torch.t(output[context,]))
        loss1 = - F.logsigmoid(data)

        data = torch.matmul(input[focus,], torch.t(output[neg_sample,]))
        loss2 = - F.logsigmoid(-data).sum()
        loss_val += loss1 + loss2
        # Pour ensuite dériver les matrices par rapport à la loss
        (loss1+loss2).backward()

        # Il faut modifier juste le .data pour ne pas perdre la structure
        input.data = input.data - learning_rate * input.grad.data
        output.data = output.data - learning_rate * output.grad.data

        input.grad.data.zero_()
        output.grad.data.zero_()
        
    with open("data/%s/Simulation_%i_seed%i/input_%i.file" %(nom_dossier, numero_simulation, seed, (len(loss_tot)+1)), "wb") as f:
        pickle.dump(input, f, pickle.HIGHEST_PROTOCOL)
    with open("data/%s/Simulation_%i_seed%i/output_%i.file" %(nom_dossier, numero_simulation, seed, (len(loss_tot)+1)), "wb") as f:
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
    with open("data/%s/Simulation_%i_seed%i/loss.file" %(nom_dossier, numero_simulation, seed), "wb") as f:
        pickle.dump(loss_tot, f, pickle.HIGHEST_PROTOCOL)
    with open("data/%s/Simulation_%i_seed%i/temps_par_epoch.file" %(nom_dossier, numero_simulation, seed), "wb") as f:
        pickle.dump(temps_par_epoch, f, pickle.HIGHEST_PROTOCOL)
        
    end_epoch = time.time()
    temps_par_epoch.append(end_epoch - start_epoch)
    loss_val = loss_val / taille_vocab
    loss_tot.append(loss_val)
    if plot:
        live_plot(loss_tot)
    print(round((end_epoch - start_epoch)/60, 2))
end = time.time()
print(round((end - start)/60, 2))
print(statistics.mean(temps_par_epoch)/60)


# In[27]:


nb_tweets = 100000
with open('data/%ik/Simulation_%ibis2/input' % int(nb_tweets/1000) + 
          "" %numero_simulation +
          "_1.file", "rb") as f:
    input2 = pickle.load(f)
with open('data/%ik/' % int(nb_tweets/1000) +
          "Simulation_%ibis2/output" %numero_simulation + 
          "_1.file", "rb") as f:
    output2 = pickle.load(f)
with open('data/%ik/' % int(nb_tweets/1000) + 
          "Simulation_%i/input" %numero_simulation +
          "_1.file", "rb") as f:
    input = pickle.load(f)
with open('data/%ik/' % int(nb_tweets/1000) +
          "Simulation_%i/output" %numero_simulation + 
          "_1.file", "rb") as f:
    output = pickle.load(f)
print(torch.all(torch.eq(input, input2)))


# In[12]:


torch.manual_seed(1)
input = torch.randn(taille_vocab, dim)
torch.manual_seed(1)
input2 = torch.randn(taille_vocab, dim)
print(torch.all(torch.eq(input, input2)))


# In[10]:


dim


# In[ ]:




