#!/usr/bin/env python
# coding: utf-8

# In[2]:


import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


# In[3]:


#base = importr('base')
dir = '/Users/alainquartierlatente/Desktop/Ensae/StatApp'
dossier = "ens"
robjects.r["setwd"]('{}/data/{}/gensim'.format(dir,dossier))
fichiers = robjects.r["list.files"](pattern = "Simulation")
fichiers


# In[4]:


len(fichiers)
#numéro de la simul
#Pour chaque simul on définit dans l'orde: dim, ep, window, learning rate. 
# Seed déterminée par num_seed
def_simul = {1:[20,0.01,3,10],2:[20,0.01,3,10], 3:[20,0.01,4,10], 4:[20,0.01,5,10],
8:[20,0.005,3,10], 9:[20,0.02,3,10], 10:[20,0.03,3,10], 11:[20,0.04,3,10], 12:[20,0.05,3,10],
13:[20,0.02,4,10],14:[20,0.02,5,10]}
num_simul = robjects.r["gsub"]("(Simulation_)|(_seed.*)","",fichiers)
num_seed = robjects.r["gsub"]("(Simulation_\d*_?)|(seed)","",fichiers)
print(num_simul)
print(num_seed)


# In[6]:


new_names = []
for i in range(len(num_simul)):
    simul = int(num_simul[i])
    #numero_simul, dim learning_Rate, window
    dim = def_simul[simul][0]
    learning_rate = def_simul[simul][1]
    window = def_simul[simul][2]
    epoch = def_simul[simul][3]
    seed = num_seed[i]
    chemin = "dim{}_ep{}_w{}_lr{}_seed{}".format(dim, epoch, window, str(learning_rate)[2:], seed)
    print("{} à {}".format(fichiers[i],chemin))
    new_names.append(chemin)


# In[53]:


robjects.StrVector(new_names)


# In[7]:


robjects.r["file.rename"](fichiers, robjects.StrVector(new_names))

