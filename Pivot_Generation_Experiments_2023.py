#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import pyterrier as pt
if not pt.started():
    pt.init()
pt.set_tqdm('tqdm')
import xml.etree.ElementTree as ET
import json
import random
import math
import os
import scipy as sp


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[11]:


get_ipython().run_line_magic('run', 'Pivot_Generation_Tools.ipynb')

from ipynb.fs.full.Pivot_Generation_Tools import *


# In[4]:


def get_performance (run, queries, qrels, measure):
    #print(run, queries, qrels)
    
    e = pt.Experiment(
        [run],
        queries,
        qrels,
        eval_metrics=[measure],
    )
    
    #print(e[measure][0])
    return e[measure][0]


# In[5]:


# Runs from other participants (official submissions)
runs_link_TRECDL  = "/home/mrim/galuscap/data/TRECDL-runs"

import os

primary_submissions_TRECDL = []

for link in os.listdir(runs_link_TRECDL):
    
    if (not ".gz" in link):
        continue
    
    full_link = runs_link_TRECDL + "/" + link
    run = pt.io.read_results(full_link)
    
    primary_submissions_TRECDL.append(run)


# In[ ]:





# In[6]:


dataset_TRECDL = pt.get_dataset("irds:msmarco-passage-v2")
queries_TRECDL = pt.io.read_topics("/home/mrim/data/collection/msmarco_v2/topics/2022_queries.trec", format='trec')
qrels_TRECDL = pt.io.read_qrels("/home/mrim/data/collection/msmarco_v2/topics/2022.qrels.pass.withDupes.no1.txt")
# Qrels without duplicates !!!


# In[ ]:


# Statistics of the submitted runs

map_TRECDL = get_runs_statistics(primary_submissions_TRECDL, queries_TRECDL, qrels_TRECDL, "map")
p10_TRECDL = get_runs_statistics(primary_submissions_TRECDL, queries_TRECDL, qrels_TRECDL, "P_10")
rr_TRECDL = get_runs_statistics(primary_submissions_TRECDL, queries_TRECDL, qrels_TRECDL, "recip_rank")
ndcg_TRECDL = get_runs_statistics(primary_submissions_TRECDL, queries_TRECDL, qrels_TRECDL, "ndcg")

print(map_TRECDL)
print(p10_TRECDL)
print(rr_TRECDL)
print(ndcg_TRECDL)


# In[9]:


# Statistics of the document collection

TRECDL_stemmed_indexpath = "/home/mrim/data/collection/msmarco_v2/index/terrier_stemmed/"
TRECDL_stemmed_indexref = pt.IndexRef.of(os.path.join(TRECDL_stemmed_indexpath, "data.properties"))
TRECDL_stemmed_index = pt.IndexFactory.of(TRECDL_stemmed_indexref)
print(TRECDL_stemmed_index.getCollectionStatistics())

# XXX Neds to be changed to contain number of docs

docids_TRECDL = []
for i in range (0, 138364198):
    docid = TRECDL_stemmed_index.getMetaIndex().getAllItems(i)[0]
    docids_TRECDL.append(docid)


# In[ ]:


# Get distributions of the submitted runs

params_relevant_TRECDL, params_total_relevant_TRECDL, params_unjudged_TRECDL = get_distributions(qrels_TRECDL, queries_TRECDL, docids_TRECDL, primary_submissions_TRECDL, [], "run")


# In[13]:


# Get distributions of the BM25 and TFIDF runs

tfidf_TRECDL = pt.BatchRetrieve(TRECDL_stemmed_indexref, wmodel="TF_IDF", properties={"termpipelines" : "Stopwords,PorterStemmer"})
bm25_TRECDL = pt.BatchRetrieve(TRECDL_stemmed_indexref, wmodel="BM25", properties={"termpipelines" : "Stopwords,PorterStemmer"})

tfidf_TRECDL_run = tfidf_TRECDL(queries_TRECDL)
bm25_TRECDL_run = bm25_TRECDL(queries_TRECDL)

pt.Experiment([tfidf_TRECDL_run, bm25_TRECDL_run], queries_TRECDL, qrels_TRECDL, eval_metrics=["map","P_10", "recip_rank", "ndcg"])

two_systems_TRECDL = [tfidf_TRECDL_run, bm25_TRECDL_run] 

params_relevant_TRECDL_2sys, params_total_relevant_TRECDL_2sys, params_unjudged_TRECDL_2sys = get_distributions(qrels_TRECDL, queries_TRECDL, docids_TRECDL, two_systems_TRECDL, [], "run")
print(params_relevant_TRECDL_2sys, params_total_relevant_TRECDL_2sys, params_unjudged_TRECDL_2sys)


# # Create the best and worst run estimate

# ### 1) First run the BM25 

# In[ ]:


### 2) Get m, c, and t estimates


# In[ ]:





# ### 3) How much better can we expect, that the runs will be?
# 
# #### 3.1) Neural Systems
# 
# DuoBERT: At least 2x better (Table 21 in BERT book)
# And Splade was even better
# 
# -> change m. It is linearly better, but the angle is ???
# 
# 
# 
# 
# #### 3.2) Non-neural Systems
# 
# Which BM25 Do You Mean? A Large-Scale Reproducibility Study of Scoring Variants: cahnges are minimal
# 
# RM3: Table 11 (Jimmy Lin BERT book): 7% better
# 
# BM25 Parameters: http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf?ref=https://githubhelp.com
# -x - up to 10%
# 

# # Applications
# 
# What we create is a simulation of the runs
# 
# We might need these:
# - to be able to test user's models
# - to be able to test cutoff methods (user models which 
