#!/usr/bin/env python
# coding: utf-8

# # Importation des données

# In[2]:


import os
import string
import re
import pandas as pd
import csv
import pickle

#os.chdir('C:/Users/torna/Documents/StatApp/StatApp')
#os.chdir('C:/Users/Kim Antunez/Documents/Projets_autres')
#os.chdir('/Users/alainquartierlatente/Desktop/Ensae/StatApp')
os.chdir('/home/aqlt/Documents/Ensae/StatApp')


# In[ ]:


df = pd.read_csv("data/sample_3.txt",sep='\n',header=None)
print(string.punctuation + "'’«»—")
def mise_en_forme_phrase (phrase):
    phrase = phrase.lower()
    # On enlève la ponctuation mais ça peut se discuter (garder les @ et #?)
    phrase = re.sub('( @[^ ]*)|(^@[^ ]*)'," nickname ", phrase) #Remplace @... par nickname
    # On enlève la ponctuation + certaines apostrophes
    phrase = phrase.translate(str.maketrans('', '', string.punctuation + "'’«»—"))
    # On enlève les passages à la ligne
    phrase = re.sub('\\n', ' ', phrase)
    # On enlève les tabulations
    phrase = re.sub('\\t', ' ', phrase)
    # On enlève les espaces multiples et les espaces à la fin des phrases
    phrase = re.sub(' +', ' ', phrase)
    phrase = re.sub(' +$', '', phrase)
    phrase = re.sub('^ +', '', phrase)
    # phrase.isalpha() # inutile
    return(phrase)

corpus = []
for index, row in df.iterrows():
    for j, column in row.iteritems():
        corpus.append(mise_en_forme_phrase(column))
print("formattage terminé")

#with open("data/sample_formate.txt", "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    for phrase in corpus:
#        writer.writerow([phrase]) 
#print("export csv terminé")

with open("data/corpus.file", "wb") as f:
    pickle.dump(corpus, f, pickle.HIGHEST_PROTOCOL)


# In[5]:


# Si on veut récup un fichier existant pour travailler
with open("data/corpus.file", "rb") as f:
    corpus = pickle.load(f) 


# ## Création corpus total

# In[6]:


phrases = [phrase.split() for phrase in corpus]
words = [item for sublist in phrases for item in sublist]
vocabulary = list(dict.fromkeys(words))
print("Nombre de mots :", len(words))
print("Taille du vocabulaire :", len(vocabulary))


# In[7]:


import nltk
import collections
from collections import Counter
fdist = nltk.FreqDist(words)
# On va enlever les mots qui apparaissent moins de 10 fois
mots_a_enlever = [w for w in vocabulary if fdist[w]<10]
mots_a_enlever_dict = Counter(mots_a_enlever)
len(mots_a_enlever)


# In[8]:


corpus_trie = [' '.join([(word if word not in mots_a_enlever_dict else "lowfrequencyword") for word in phrase.split()]) for phrase in corpus]


# In[9]:


# On ne garde que les tweets de 2 mots ou plus
corpus_trie = [phrase for phrase in corpus_trie if len(phrase.split())>=2]
with open("data/corpus_trieens.file", "wb") as f:
    pickle.dump(corpus_trie, f, pickle.HIGHEST_PROTOCOL)


# ## Création corpus 10k

# In[10]:


phrases = [phrase.split() for phrase in corpus[0:10000]]
words = [item for sublist in phrases for item in sublist]
vocabulary = list(dict.fromkeys(words))
print("Nombre de mots :", len(words))
print("Taille du vocabulaire :", len(vocabulary))


# In[11]:


import nltk
import collections
from collections import Counter
fdist = nltk.FreqDist(words)
# On va enlever les mots qui apparaissent moins de 3 fois
mots_a_enlever = [w for w in vocabulary if fdist[w]<3]
mots_a_enlever_dict = Counter(mots_a_enlever)
len(mots_a_enlever)


# In[12]:


corpus_trie10k = [' '.join([(word if word not in mots_a_enlever_dict else "lowfrequencyword") for word in phrase.split()]) for phrase in corpus[0:10000]]


# In[13]:


# On ne garde que les tweets de 2 mots ou plus
corpus_trie10k = [phrase for phrase in corpus_trie10k if len(phrase.split())>=2]
with open("data/corpus_trie10k.file", "wb") as f:
    pickle.dump(corpus_trie10k, f, pickle.HIGHEST_PROTOCOL)


# ## Création corpus 100k

# In[14]:


phrases = [phrase.split() for phrase in corpus[0:100000]]
words = [item for sublist in phrases for item in sublist]
vocabulary = list(dict.fromkeys(words))
print("Nombre de mots :", len(words))
print("Taille du vocabulaire :", len(vocabulary))


# In[15]:


import nltk
import collections
from collections import Counter
fdist = nltk.FreqDist(words)
# On va enlever les mots qui apparaissent moins de 5 fois
mots_a_enlever = [w for w in vocabulary if fdist[w]<5]
mots_a_enlever_dict = Counter(mots_a_enlever)
len(mots_a_enlever)


# In[16]:


corpus_trie100k = [' '.join([(word if word not in mots_a_enlever_dict else "lowfrequencyword") for word in phrase.split()]) for phrase in corpus[0:100000]]


# In[17]:


# On ne garde que les tweets de 2 mots ou plus
corpus_trie100k = [phrase for phrase in corpus_trie100k if len(phrase.split())>=2]
with open("data/corpus_trie100k.file", "wb") as f:
    pickle.dump(corpus_trie100k, f, pickle.HIGHEST_PROTOCOL)


# ## Vocabulaire après traitement

# In[18]:


phrases = [phrase.split() for phrase in corpus_trie]
phrases10k = [phrase.split() for phrase in corpus_trie10k]
phrases100k = [phrase.split() for phrase in corpus_trie100k]
words = [item for sublist in phrases for item in sublist]
words10k = [item for sublist in phrases10k for item in sublist]
words100k = [item for sublist in phrases100k for item in sublist]
vocabulary = list(dict.fromkeys(words))
vocabulary10k = list(dict.fromkeys(words10k))
vocabulary100k = list(dict.fromkeys(words100k))
print("Sur 10k")
print("Nombre de mots :", len(words10k))
print("Taille du vocabulaire :", len(vocabulary10k))
print("Sur 100k")
print("Nombre de mots :", len(words100k))
print("Taille du vocabulaire :", len(vocabulary100k))
print("Sur tout")
print("Nombre de mots :", len(words))
print("Taille du vocabulaire :", len(vocabulary))

