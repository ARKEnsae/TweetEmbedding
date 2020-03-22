#!/usr/bin/env python
# coding: utf-8

# # Importation des données

# In[38]:


import os
import string
import re
import math
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
#useless ?

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import pickle

#PCA & TSNE : 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import seaborn as sns

import gensim
from gensim.models import word2vec

import rpy2.robjects as r


# In[2]:


nom_dossier = "100k"
#nom_dossier = "ens"

gensim_utile = True #True pour le modèle google, False pour le modele ARK

#os.chdir('C:/Users/torna/Documents/StatApp/StatApp')
#os.chdir('/Users/alainquartierlatente/Desktop/Ensae/StatApp')
os.chdir('C:/Users/Kim Antunez/Documents/Projets_autres/StatApp')
#os.chdir('/home/aqlt/Documents/Ensae/StatApp')


with open("data/corpus_trie%s.file" %nom_dossier, "rb") as f:
    corpus = pickle.load(f) 
ens_tweets = [phrase.split() for phrase in corpus]
len(ens_tweets)


# On crée index_mot et mot_index

# In[3]:


phrases = ens_tweets
words = [item for sublist in phrases for item in sublist]
with open('data/%s/vocabulary.file' %nom_dossier, "rb") as f:
    vocabulary = pickle.load(f) 
index_mot = {index: w for (index, w) in enumerate(vocabulary)}
mot_index = {w: index for (index, w) in enumerate(vocabulary)}
fdist = nltk.FreqDist(words)


# # Fonctions générales
# 
# ## 1. Distances
# 
# Pour tester la similarité entre deux mots, on calcule la **similarité cosinus** entre leur représentation vectorielle :
# $$
# \frac{A \cdot B}{|| A || \cdot || B ||}
# $$
# 
# On peut également calculer la **distance euclidienne** pour comparer les résultats avec ces deux métriques

# In[14]:


def cos_distance(u, v):
    return (np.dot(u, v)  / (math.sqrt(np.dot(u, u)) *  (math.sqrt(np.dot(v, v)))))

#def eucl_distance(u, v):
#    return (1/(np.linalg.norm(u/np.dot(u, u)-v/np.dot(v, v)))+1)
def eucl_distance(u, v):
    return (-np.linalg.norm(u/np.dot(u, u)-v/np.dot(v, v)))



# Cette fonction permet d'afficher les n mots les plus proche d'un mot (distance cosinus ou euclidienne)

# In[15]:


def mot_plus_proche(word, n=10, distance="cos"):
    word_distance = {}
    for mot in mot_poids:
        if mot != word:
            if distance == "cos":
                word_distance[mot] = (cos_distance(mot_poids[mot],(mot_poids[word])))
            else:
                word_distance[mot] = (eucl_distance(mot_poids[mot],(mot_poids[word])))
    word_distance = sorted(word_distance.items(), key=lambda t: t[1],reverse=True)
    return word_distance[0:n]


# Cette fonction permet de calculer la distance entre deux mots. 

# In[16]:


def distance_mots(word1,word2, mot_poids, distance="cos"):
    if word1 in mot_poids.keys() and word2 in mot_poids.keys():
        if(distance=="cos"):
            word_distance = (cos_distance(mot_poids[word1],(mot_poids[word2])))
        else:
            word_distance = (eucl_distance(mot_poids[word1],(mot_poids[word2])))
    else:
        word_distance = float('nan')
    return word_distance
distance_mots_v = np.vectorize(distance_mots)


# # Evaluer plusieurs modèles 
# 
# ## 1. Evaluation par le "human judgement"
# 

# In[28]:


def corr_modele(chemin, distance="cos", gensimbool = gensim_utile, epoch_max = 50):  
    all_cor = []
    all_p_val = []
    if gensimbool:
        #chemin = "data/" + nom_dossier + "/gensim/Simulation_" + str(numero_simulation) + dossier
        #chemin = "data/{}/gensim/dim{}_ep{}_w{}_lr{}_seed{}".format(nom_dossier, dim, epoch, window, str(learning_rate)[2:], seed)  
        isdir = os.path.isdir(chemin)
        if isdir:
            model = gensim.models.keyedvectors.KeyedVectors.load(chemin + "/word2vec.model")
            py_list = []
            for mot in vocabulary:
                py_list.append(model.wv[mot].tolist())   
            mat_finale = torch.FloatTensor(py_list)
            mot_poids = {index_mot[index]: poids.detach().numpy() for (index, poids) in enumerate(mat_finale)}
            df = pd.read_csv('data_bis/word_similarity.csv', sep=";")
            df["corr_word2vec"] = distance_mots_v(df["word1"],df["word2"], mot_poids, distance)
            df = df.dropna()
            corr, p_value = spearmanr(df["corr"], df["corr_word2vec"])
            all_cor.append(corr)
            all_p_val.append(p_value)
    else:
        isdir = os.path.isdir(chemin)
        if isdir:
            for i in range(epoch_max):
                try:
                    with open('{}/input_{}.file'.format(chemin, i+1), "rb") as f:
                        input = pickle.load(f)
                    with open('{}/output{}.file'.format(chemin, i+1), "rb") as f:
                        output = pickle.load(f)
                    mat_finale = (input+output)/2
                    mot_poids = {index_mot[index]: poids.detach().numpy() for (index, poids) in enumerate(mat_finale)}
                    df = pd.read_csv('data_bis/word_similarity.csv', sep=";")
                    df["corr_word2vec"] = distance_mots_v(df["word1"],df["word2"], mot_poids, distance)
                    df = df.dropna()
                    corr, p_value = spearmanr(df["corr"], df["corr_word2vec"])
                    all_cor.append(corr)
                    all_p_val.append(p_value)
                except OSError:
                    break;     
        
    return([all_cor, all_p_val]);

#corr_modele("data/100k/gensim/dim20_ep10_w3_lr01_seed1", distance="cos", gensimbool = True, epoch_max = 50)


def tableau_corr_hji(dim = "\d*", epoch = "\d*", window = "\d*",
                     learning_rate = "\d*", seed = "\d*",
                     distance="cos", index = "seed", gensimbool = gensim_utile):    
    if(isinstance(learning_rate, float)):
        learning_rate = str(learning_rate)[2:]
    if(isinstance(dim, list)):
        dim = r.r['paste0']("(",r.r['paste0']("(",dim,")",collapse = "|"),")")[0]
    if(isinstance(epoch, list)):
        epoch = r.r['paste0']("(",r.r['paste0']("(",epoch,")",collapse = "|"),")")[0]
    if(isinstance(window, list)):
        window = r.r['paste0']("(",r.r['paste0']("(",window,")",collapse = "|"),")")[0]
    if(isinstance(learning_rate, list)):
        learning_rate = [str(lr)[2:] if isinstance(lr, float) else lr for lr in learning_rate]
        learning_rate = r.r['paste0']("(",r.r['paste0']("(",learning_rate,")",collapse = "|"),")")[0]
    if(isinstance(seed, list)):
        seed = r.r['paste0']("(",r.r['paste0']("(",seed,")",collapse = "|"),")")[0]

    requete_regexp = 'dim{}_ep{}_w{}_lr{}_seed{}'.format(dim, epoch, window, learning_rate, seed)
    if gensimbool:
        gensim_st = "gensim/"
    else:
        gensim_st = ""
    dossiers = r.r['list.files'](path = 'data/{}/{}'.format(nom_dossier, gensim_st),
                      pattern = requete_regexp)
    dossiers = [dossiers[i] for i in range(len(dossiers))]
    #dossiers = r.r['paste0']('data/{}/{}'.format(nom_dossier, gensim_st),dossiers)
    
    resultats = []
    nom_cols = dossiers

    for dossier in dossiers:
        #nom_cols.append( "dim{}_ep{}_lr{}_w{}".format(simul[0], simul[1], simul[2], str(simul[3])[2:]))
        #print(nom_cols)
        resultats.append(corr_modele(chemin = 'data/{}/{}{}'.format(nom_dossier, gensim_st, dossier), 
                                 distance= distance,gensimbool = gensimbool))
    max_len = max([len(x[0]) for x in resultats])
    for i in range(len(resultats)):
        if(len(resultats[i][0])<max_len):
            resultats[i][0] = resultats[i][0] +[np.nan]*(max_len - len(resultats[i][0]))
            resultats[i][1] = resultats[i][1] +[np.nan]*(max_len - len(resultats[i][1]))
    #nom_lignes = [re.sub('_seed', 'Seed ', d) if d != "" else "Pas de seed" for d in dossiers ]
    #nom_lignes = ["Seed " + str(s) for s in seeds]
    dossiers_sep = [string.split('_') for string in dossiers]
    all_dim = [r.r['grep']("^dim", x, value = True)[0] for x in dossiers_sep]
    all_dim = [r.r['paste0']("dim=", r.r['gsub']("dim", "", x))[0] for x in all_dim] 
    all_ep = [r.r['grep']("^ep", x, value = True)[0] for x in dossiers_sep]
    all_ep = [r.r['paste0']("ep=", r.r['gsub']("ep", "", x))[0] for x in all_ep] 
    all_w = [r.r['grep']("^w", x, value = True)[0] for x in dossiers_sep]
    all_w = [r.r['paste0']("w=", r.r['gsub']("w", "", x))[0] for x in all_w] 
    all_lr = [r.r['grep']("^lr", x, value = True)[0] for x in dossiers_sep]
    all_lr = [r.r['paste0']("lr=0.", r.r['gsub']("lr", "", x))[0] for x in all_lr] 
    all_seed = [r.r['grep']("^seed", x, value = True)[0] for x in dossiers_sep]
    all_seed = [r.r['paste0']("seed=", r.r['gsub']("seed", "", x))[0] for x in all_seed] 
    
    all_name = [all_dim, all_ep, all_w, all_lr, all_seed]


    #all_ep = r.r['grep']("^ep", x, value = True)
    #all_ep = [all_ep[0] for x in dossiers_sep]
    #all_w = r.r['grep']("^w", x, value = True)
    #all_w = [all_w[0] for x in dossiers_sep]
    #all_lr = [r.r['grep']("^lr", x, value = True)[0] for x in dossiers_sep]
    #all_seed = [r.r['grep']("^seed", x, value = True)[0] for x in dossiers_sep]
    #lr=0.01\n w=3\n e=10
    
    correspondance = {'dim': 0,
                  'epoch': 1,
                  'window': 2,
                  'learning_rate': 3,
                 'seed': 4}
    if not isinstance(index, list):
        index = [index]
    index_num = [correspondance[i] for i in index]
    indexs = [all_name[i] for i in [0,1,2,3,4] if i in index_num ]
    column = [all_name[i] for i in [0,1,2,3,4] if i not in index_num]
    lignes = list(map("\n".join, zip(*indexs)))
    colonnes = list(map("\n".join, zip(*column)))
    pivot_df = pd.DataFrame({'lignes': lignes,
                  'colonnes': colonnes})
    
    all_cor = pd.DataFrame([elem[0] for elem in resultats])
    all_p_val = pd.DataFrame([elem[1] for elem in resultats])
    all_cor.columns = all_p_val.columns = ["values"]
    

    all_cor = pd.concat([pivot_df, all_cor], axis=1)
    all_cor = all_cor.pivot(index = "lignes", columns = "colonnes", values = 'values')
    all_p_val = pd.concat([pivot_df, all_p_val], axis=1)
    all_p_val = all_p_val.pivot(index = "lignes", columns = "colonnes", values = 'values')
    
    # Pour enlever les noms ajouté par la fonction pivot
    all_cor = all_cor.rename_axis(None, axis=1)  
    all_cor = all_cor.rename_axis(None, axis=0)
    all_p_val = all_p_val.rename_axis(None, axis=1)  
    all_p_val = all_p_val.rename_axis(None, axis=0)
    return([all_cor, all_p_val])


# In[41]:


# Explication : on peut fixer ou nom certains paramètres : par défaut on va chercher tous les dossiers disponibles
# Par exemple si on ne met pas le paramètre learning_rate, le programme va cherche 0.01 et 0.02 (s'ils existent)
# le paramètre index permet de définir les paramètres que l'on veut en ligne. Si l'on souhaite en mettre 2 faire une liste
# par exemple ["seed", "dim"]

#resultats_hji = tableau_corr_hji(learning_rate = [0.01,0.02],
#                       window = [3,4,5], dim = 50, index="seed",
#                       distance = "cos")[0]
# Pour mettre dans l'ordre les lignes
#resultats_hji = resultats_hji.reindex(index = ['seed=1','seed=5','seed=10','seed=15','seed=20','seed=25'])

##Il faut installer pandas 1.0 et tabulate
#print(resultats_hji)
#print(resultats_hji.to_markdown())


# In[70]:


from matplotlib.lines import Line2D

def intervalles_confiance (resultats_hji, distance = "cos"):
    colors = sns.color_palette("hls", 6)
    fig = plt.figure(figsize = (12,6))    
    ax = fig.add_axes([0.1, 0.1, 1, 1])
    corr = resultats_hji
    corr = corr.reindex(index = ['seed=1','seed=5','seed=10','seed=15','seed=20','seed=25'])
    # ax.set_ylim(-0.5,0.6) # seulement si besoin pour faire des graph comparables
    ax.set_xlim(-0.2,len(corr.columns)-1+.2)

    for i in range(len(corr.columns)):
        corr_values = corr.iloc[:,i].values.tolist()
        ec = np.std(corr_values)
        moy = np.mean(corr_values)
        #print(moy)
        icm = moy - 1.96 * ec
        icp = moy + 1.96 * ec
        ax.errorbar(i, moy, yerr=1.96 * ec,ecolor = "black", alpha = 0.5) 
        ax.scatter([i]*len(corr_values), corr_values, c=colors, s = 100) 
        ax.errorbar(i, moy, xerr=0.1, ecolor = "black", alpha= 0.5)
        
    ax.set_xticks(list(range(len(corr.columns))))
    #ax.set_xticklabels(["Simul. n°%i" %simul[i] for i in list(range(len(simul)))])
    ax.set_xticklabels(corr.columns.values.tolist())
    #ax.set_xticklabels(["lr=0.01\n w=3\n e=10","lr=0.01\n w=4\n e=10","lr=0.01\n w=5\n e=10",
    #                    "lr=0.02\n w=3\n e=10","lr=0.02\n w=4\n e=10","lr=0.02\n w=5\n e=10",
    #                    "lr=0.01\n w=3\n e=100","lr=0.01\n w=4\n e=100","lr=0.01\n w=5\n e=100",
    #                    "lr=0.02\n w=3\n e=100","lr=0.02\n w=4\n e=100","lr=0.02\n w=5\n e=100"])
    plt.ylabel("Corrélation entre la distance " + distance + " et le l'indice de jugement humain")
    plt.xlabel("Simulation avec "+ nom_dossier +" tweets")
    custom_lines=[]
    for i in range(len(colors)):
        custom_lines.append(Line2D([0], [0], marker='o', color='w',
                                   label='Scatter',markerfacecolor=colors[i], markersize=10))
    ax.legend(custom_lines, ['Seed 1','Seed 5','Seed 10','Seed 15','Seed 20','Seed 25'])


# In[71]:


#intervalles_confiance(resultats_hji)


######################## Graphiques

res_100k = tableau_corr_hji(learning_rate = [0.01,0.02],
                       window = [3,4,5], dim = 50, index="seed",
                       distance = "cos", epoch=[10,100])[0]

