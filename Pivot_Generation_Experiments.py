#!/usr/bin/env python
# coding: utf-8

# # Aquaint

# In[3]:


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


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


get_ipython().run_line_magic('run', 'Pivot_Generation_Tools.ipynb')

from ipynb.fs.full.Pivot_Generation_Tools import *


# In[ ]:


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


runs_link_aquaint  = "/home/mrim/galuscap/data/AQUAINT-runs"

import os

primary_submissions_aquaint = []

for link in os.listdir(runs_link_aquaint):
    
    if (not ".gz" in link):
        continue
    
    full_link = runs_link_aquaint + "/" + link
    run = pt.io.read_results(full_link)
    
    primary_submissions_aquaint.append(run)


# In[6]:


dataset_aquaint = pt.get_dataset("irds:aquaint/trec-robust-2005")
queries_aquaint = dataset_aquaint.get_topics("title")
qrels_aquaint = dataset_aquaint.get_qrels()


# In[7]:


max_aquaint_map = get_maximum_score(qrels_aquaint, queries_aquaint, "map")
print(max_aquaint_map)


# In[8]:


#print (primary_submissions_aquaint)

map_aquaint = get_runs_statistics(primary_submissions_aquaint, queries_aquaint, qrels_aquaint, "map")
p10_aquaint = get_runs_statistics(primary_submissions_aquaint, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint = get_runs_statistics(primary_submissions_aquaint, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint = get_runs_statistics(primary_submissions_aquaint, queries_aquaint, qrels_aquaint, "ndcg")

print(map_aquaint)
print(p10_aquaint)
print(rr_aquaint)
print(ndcg_aquaint)


# In[10]:


aquaint_indexpath = "/home/mrim/galuscap/data/indexes/AQUAINT"
aquaint_indexref = pt.IndexRef.of(os.path.join(aquaint_indexpath, "data.properties"))
aquaint_index = pt.IndexFactory.of(aquaint_indexref)
print(aquaint_index.getCollectionStatistics())

docids_aquaint = []
for i in range (0, 1033461):
    docid = aquaint_index.getMetaIndex().getAllItems(i)[0]
    docids_aquaint.append(docid)


# In[ ]:


params_relevant_aquaint, params_total_relevant_aquaint, params_unjudged_aquaint = get_distributions(qrels_aquaint, queries_aquaint, docids_aquaint, primary_submissions_aquaint, [], "run")


# In[ ]:


pivots_expo_total_unjudged_aquaint_item, minscore, maxscore = generate_pivots(qrels_aquaint, queries_aquaint, docids_aquaint, params_total_relevant_aquaint, params_unjudged_aquaint, 1000, True, "map", method = "exponential_unjudged", bin_samples = 10, bin_size = 0.1, max_runs = 10 )


# In[ ]:


map_aquaint_pivots = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "map")
p10_aquaint_pivots = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint_pivots = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint_pivots  = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "ndcg")

print(map_aquaint_pivots)
print(p10_aquaint_pivots)
print(rr_aquaint_pivots)
print(ndcg_aquaint_pivots)


# In[ ]:


get_distributions(qrels_aquaint, queries_aquaint, docids_aquaint, pivots_expo_total_unjudged_aquaint_item, [], "pivot")


# In[ ]:


pivots_expo_total_unjudged_aquaint_distorted_item, minscore, maxscore = generate_pivots(qrels_aquaint, queries_aquaint, docids_aquaint, params_total_relevant_aquaint, params_unjudged_aquaint, 1000, True, "map", method = "exponential_unjudged_distorted", bin_samples = 10, bin_size = 0.1, max_runs = 10 )


# In[ ]:


map_aquaint_pivots = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_distorted_item, queries_aquaint, qrels_aquaint, "map")
p10_aquaint_pivots = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_distorted_item, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint_pivots = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_distorted_item, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint_pivots  = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_distorted_item, queries_aquaint, qrels_aquaint, "ndcg")

print(map_aquaint_pivots)
print(p10_aquaint_pivots)
print(rr_aquaint_pivots)
print(ndcg_aquaint_pivots)


# In[ ]:


get_distributions(qrels_aquaint, queries_aquaint, docids_aquaint, pivots_expo_total_unjudged_aquaint_distorted_item, [], "pivot")


# In[ ]:





# In[11]:


tfidf_aquaint = pt.BatchRetrieve(aquaint_indexref, wmodel="TF_IDF")
bm25_aquaint = pt.BatchRetrieve(aquaint_indexref, wmodel="BM25")

tfidf_aquaint_run = tfidf_aquaint(queries_aquaint)
bm25_aquaint_run = bm25_aquaint(queries_aquaint)

pt.Experiment([tfidf_aquaint, bm25_aquaint], queries_aquaint, qrels_aquaint, eval_metrics=["map","P_10", "recip_rank", "ndcg"])

two_systems_aquaint = [tfidf_aquaint_run, bm25_aquaint_run]


# In[13]:


params_relevant_aquaint_2sys, params_total_relevant_aquaint_2sys, params_unjudged_aquaint_2sys = get_distributions(qrels_aquaint, queries_aquaint, docids_aquaint, two_systems_aquaint, [], "run")


# In[16]:


print(params_total_relevant_aquaint_2sys)


# In[18]:


map_aquaint_2sys = get_runs_statistics(two_systems_aquaint,  queries_aquaint, qrels_aquaint, "map")
p10_aquaint_2sys = get_runs_statistics(two_systems_aquaint, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint_2sys = get_runs_statistics(two_systems_aquaint, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint_2sys  = get_runs_statistics(two_systems_aquaint, queries_aquaint, qrels_aquaint, "ndcg")


print(map_aquaint_2sys)
print(p10_aquaint_2sys)
print(rr_aquaint_2sys)
print(ndcg_aquaint_2sys)


# In[14]:


pivots_expo_total_unjudged_aquaint_2sys, minscore, maxscore = generate_pivots(qrels_aquaint, queries_aquaint, docids_aquaint, params_total_relevant_aquaint_2sys, params_unjudged_aquaint_2sys, 1000, True, "map", method = "exponential_unjudged", bin_samples = 10, bin_size = 0.1, max_runs = 10 )


# In[15]:


get_distributions(qrels_aquaint, queries_aquaint, docids_aquaint, pivots_expo_total_unjudged_aquaint_2sys, [], "pivot")


# In[19]:


map_aquaint_2sys_pivot = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys,  queries_aquaint, qrels_aquaint, "map")
p10_aquaint_2sys_pivot = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint_2sys_pivot = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint_2sys_pivot  = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys, queries_aquaint, qrels_aquaint, "ndcg")


print(map_aquaint_2sys_pivot)
print(p10_aquaint_2sys_pivot)
print(rr_aquaint_2sys_pivot)
print(ndcg_aquaint_2sys_pivot)


# In[34]:


pivots_expo_total_unjudged_aquaint_2sys_judgedonly, minscore, maxscore = generate_pivots(qrels_aquaint, queries_aquaint, docids_aquaint, params_relevant_aquaint_2sys, params_unjudged_aquaint_2sys, 1000, True, "map", method = "exponential", bin_samples = 10, bin_size = 0.1, max_runs = 10 )


# In[35]:


get_distributions(qrels_aquaint, queries_aquaint, docids_aquaint, pivots_expo_total_unjudged_aquaint_2sys_judgedonly, [], "pivot")


# In[36]:


map_aquaint_2sys_pivot_judgedonly = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys_judgedonly,  queries_aquaint, qrels_aquaint, "map")
p10_aquaint_2sys_pivot_judgedonly = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys_judgedonly, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint_2sys_pivot_judgedonly = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys_judgedonly, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint_2sys_pivot_judgedonly  = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_2sys_judgedonly, queries_aquaint, qrels_aquaint, "ndcg")


print(map_aquaint_2sys_pivot_judgedonly)
print(p10_aquaint_2sys_pivot_judgedonly)
print(rr_aquaint_2sys_pivot_judgedonly)
print(ndcg_aquaint_2sys_pivot_judgedonly)


# # WT10g

# In[20]:


runs_link_wt10g  = "/home/mrim/galuscap/data/wt10g-runs"

import os

primary_submissions_wt10g = []

for link in os.listdir(runs_link_wt10g):
    
    if (not ".gz" in link):
        continue
    
    full_link = runs_link_wt10g + "/" + link
    run = pt.io.read_results(full_link)
    
    primary_submissions_wt10g.append(run)


# In[21]:


dataset_wt10g = pt.get_dataset("trec-wt10g")
queries_wt10g = dataset_wt10g.get_topics("trec9")
qrels_wt10g = dataset_wt10g.get_qrels("trec9")


# In[ ]:


print(queries_wt10g)


# In[ ]:


map_wt10g = get_runs_statistics(primary_submissions_wt10g, queries_wt10g, qrels_wt10g, "map")
p10_wt10g = get_runs_statistics(primary_submissions_wt10g, queries_wt10g, qrels_wt10g, "P_10")
rr_wt10g = get_runs_statistics(primary_submissions_wt10g, queries_wt10g, qrels_wt10g, "recip_rank")
ndcg_wt10g = get_runs_statistics(primary_submissions_wt10g, queries_wt10g, qrels_wt10g, "ndcg")

print(map_wt10g)
print(p10_wt10g)
print(rr_wt10g)
print(ndcg_wt10g)


# In[22]:


wt10g_indexpath = "/home/mrim/galuscap/data/indexes/wt10g_index"
wt10g_indexref = pt.IndexRef.of(os.path.join(wt10g_indexpath, "data.properties"))
wt10g_index = pt.IndexFactory.of(wt10g_indexref)
print(wt10g_index.getCollectionStatistics())

docids_wt10g = []
for i in range (0, 1692096):
    docid = wt10g_index.getMetaIndex().getAllItems(i)[0]
    docids_wt10g.append(docid)


# In[ ]:


for query in queries_wt10g["qid"]:
    query_qrel = qrels_wt10g[qrels_wt10g["qid"] == query]
    positive_query_qrels = len(query_qrel[query_qrel["label"] > 0])
    print (len(query_qrel), positive_query_qrels)


# In[ ]:


params_relevant_wt10g, params_total_relevant_wt10g, params_unjudged_wt10g = get_distributions(qrels_wt10g, queries_wt10g, docids_wt10g, primary_submissions_wt10g, [], "run")


# In[ ]:


pivots_expo_total_unjudged_wt10g_item, minscore, maxscore = generate_pivots(qrels_wt10g, queries_wt10g, docids_wt10g, params_total_relevant_wt10g, params_unjudged_wt10g, 1000, True, "map", method = "exponential_unjudged", bin_samples = 10, bin_size = 0.1, max_runs = 10 )


# In[ ]:


def get_pivot_statistics(pivots, queries, qrels, measure):
    
    scores = []
    
    print("here")
    
    print(len(pivots))
    
    i = 0
    while i <= 1000:
        
        if (i in pivots):
        
            b = 0

            while b < len(pivots[i]):

                pivot = pivots[i][b]
                score = get_performance(pivot, queries, qrels, measure)
                print(score)

                scores.append(score)

                b = b+1        

        i = i + 1
        
    return(statistics.mean(scores), statistics.stdev(scores))


# In[ ]:


map_wt10g_pivots = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_item, queries_wt10g, qrels_wt10g, "map")
p10_wt10g_pivots = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_item, queries_wt10g, qrels_wt10g, "P_10")
rr_wt10g_pivots = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_item, queries_wt10g, qrels_wt10g, "recip_rank")
ndcg_wt10g_pivots  = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_item, queries_wt10g, qrels_wt10g, "ndcg")

print(map_wt10g_pivots)
print(p10_wt10g_pivots)
print(rr_wt10g_pivots)
print(ndcg_wt10g_pivots)


# In[ ]:


get_distributions(qrels_wt10g, queries_wt10g, docids_wt10g, pivots_expo_total_unjudged_wt10g_item, [], "pivot")


# In[ ]:





# In[27]:


tfidf_wt10g = pt.BatchRetrieve(wt10g_indexref, wmodel="TF_IDF")
bm25_wt10g = pt.BatchRetrieve(wt10g_indexref, wmodel="BM25")

tfidf_wt10g_run = tfidf_wt10g(queries_wt10g)
bm25_wt10g_run = bm25_wt10g(queries_wt10g)

pt.Experiment([tfidf_wt10g, bm25_wt10g], queries_wt10g, qrels_wt10g, eval_metrics=["map","P_10", "recip_rank", "ndcg"])

two_systems_wt10g = [tfidf_wt10g_run, bm25_wt10g_run]


# In[28]:


params_relevant_wt10g_2sys, params_total_relevant_wt10g_2sys, params_unjudged_wt10g_2sys = get_distributions(qrels_wt10g, queries_wt10g, docids_wt10g, two_systems_wt10g, [], "run")


# In[29]:


print(params_total_relevant_wt10g_2sys)


# In[32]:


map_wt10g_2sys = get_runs_statistics(two_systems_wt10g,  queries_wt10g, qrels_wt10g, "map")
p10_wt10g_2sys = get_runs_statistics(two_systems_wt10g, queries_wt10g, qrels_wt10g, "P_10")
rr_wt10g_2sys = get_runs_statistics(two_systems_wt10g, queries_wt10g, qrels_wt10g, "recip_rank")
ndcg_wt10g_2sys  = get_runs_statistics(two_systems_wt10g, queries_wt10g, qrels_wt10g, "ndcg")


print(map_wt10g_2sys)
print(p10_wt10g_2sys)
print(rr_wt10g_2sys)
print(ndcg_wt10g_2sys)


# In[30]:


pivots_expo_total_unjudged_wt10g_2sys, minscore, maxscore = generate_pivots(qrels_wt10g, queries_wt10g, docids_wt10g, params_total_relevant_wt10g_2sys, params_unjudged_wt10g_2sys, 1000, True, "map", method = "exponential_unjudged", bin_samples = 10, bin_size = 0.1, max_runs = 10 )


# In[31]:


get_distributions(qrels_wt10g, queries_wt10g, docids_wt10g, pivots_expo_total_unjudged_wt10g_2sys, [], "pivot")


# In[33]:


map_wt10g_2sys_pivot = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_2sys,  queries_wt10g, qrels_wt10g, "map")
p10_wt10g_2sys_pivot = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_2sys, queries_wt10g, qrels_wt10g, "P_10")
rr_wt10g_2sys_pivot = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_2sys, queries_wt10g, qrels_wt10g, "recip_rank")
ndcg_wt10g_2sys_pivot  = get_pivot_statistics(pivots_expo_total_unjudged_wt10g_2sys, queries_wt10g, qrels_wt10g, "ndcg")


print(map_wt10g_2sys_pivot)
print(p10_wt10g_2sys_pivot)
print(rr_wt10g_2sys_pivot)
print(ndcg_wt10g_2sys_pivot)


# # Per Query Results

# In[ ]:


import statistics
from distfit import distfit

normalized_means = []

with open("/home/mrim/galuscap/data/per-query/Means.csv") as file:
    for mean in file:
        normalized_means.append(float(mean))


# In[ ]:


bins_limits = np.arange(0, 1, 0.005)

print(bins_limits)

histr, binsr = np.histogram(normalized_means, bins_limits)

distr = distfit(todf=True)
normalized_means_array = np.array(histr, dtype=np.int)
distr.fit_transform(normalized_means_array)


# In[ ]:


print(histr)


# In[ ]:


distr.plot_summary()
print(distr.summary)
distr.plot()

distr_norm = distfit(distr='beta')
distr_norm.fit_transform(normalized_means_array)
distr_norm.plot()


# In[ ]:





# In[40]:


parameters_relevant_total_aquaint_test_set_t = [
    {"m": 0.35, "t": 0, "c": 0.03},
    {"m": 0.35, "t": 0.001, "c": 0.03},
    {"m": 0.35, "t": 0.002, "c": 0.03},
    {"m": 0.35, "t": 0.003, "c": 0.03},
    {"m": 0.35, "t": 0.004, "c": 0.03},
    {"m": 0.35, "t": 0.005, "c": 0.03},
    {"m": 0.35, "t": 0.008, "c": 0.03},
    {"m": 0.35, "t": 0.009, "c": 0.03},
    {"m": 0.35, "t": 0.01, "c": 0.03},
    {"m": 0.35, "t": 0.011, "c": 0.03},
    {"m": 0.35, "t": 0.015, "c": 0.03},
    {"m": 0.35, "t": 0.02, "c": 0.03},
    {"m": 0.35, "t": 0.03, "c": 0.03},
    {"m": 0.35, "t": 0.05, "c": 0.03},
    {"m": 0.35, "t": 0.1, "c": 0.03},
    {"m": 0.35, "t": 0.2, "c": 0.03},
    {"m": 0.35, "t": 0.25, "c": 0.03},
    {"m": 0.35, "t": 0.5, "c": 0.03},
    {"m": 0.35, "t": 1, "c": 0.03},
]

for parameters_relevant_total_aquaint_item in parameters_relevant_total_aquaint_test_set_t:
    
    pivots_expo_total_unjudged_aquaint_item, minscore, maxscore = generate_pivots(qrels_aquaint, queries_aquaint, docids_aquaint, parameters_relevant_total_aquaint_item, params_unjudged_aquaint_2sys, 1000, True, "map", method = "exponential_unjudged", bin_samples = 10, bin_size = 0.1, max_runs = 10 )

    print("T")
    print(parameters_relevant_total_aquaint_item["t"])
    
    print("P10")
    p_10 = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "P_10")
    print(p_10)
    print("RR")
    rr = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "recip_rank")
    print(rr)
    print("MAP")
    map = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "map")
    print(map)
    print("NDCG")
    ndcg =get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "ndcg")
    print(ndcg)


# In[42]:


parameters_relevant_total_aquaint_test_set_c = [
#    {"m": 0.35, "t": 0.01, "c": 0},
#    {"m": 0.35, "t": 0.01, "c": 0.001},
#    {"m": 0.35, "t": 0.01, "c": 0.005},
#    {"m": 0.35, "t": 0.01, "c": 0.01},
#    {"m": 0.35, "t": 0.01, "c": 0.02},
#    {"m": 0.35, "t": 0.01, "c": 0.03},
#    {"m": 0.35, "t": 0.01, "c": 0.04},
#    {"m": 0.35, "t": 0.01, "c": 0.05},
#    {"m": 0.35, "t": 0.01, "c": 0.1},
#    {"m": 0.35, "t": 0.01, "c": 0.15},
#    {"m": 0.35, "t": 0.01, "c": 0.2},
#    {"m": 0.35, "t": 0.01, "c": 0.25},
#    {"m": 0.35, "t": 0.01, "c": 0.3},
#    {"m": 0.35, "t": 0.01, "c": 0.35},
#    {"m": 0.35, "t": 0.01, "c": 0.4},
#    {"m": 0.35, "t": 0.01, "c": 0.45},
#    {"m": 0.35, "t": 0.01, "c": 0.5},
#    {"m": 0.35, "t": 0.01, "c": 0.6},
#    {"m": 0.35, "t": 0.01, "c": 0.7},
    {"m": 0.35, "t": 0.01, "c": -0.05},
    {"m": 0.35, "t": 0.01, "c": -0.1},
    {"m": 0.35, "t": 0.01, "c": -0.15},
    {"m": 0.35, "t": 0.01, "c": -0.2},
    {"m": 0.35, "t": 0.01, "c": -0.25},    
    {"m": 0.35, "t": 0.01, "c": -0.3},
    {"m": 0.35, "t": 0.01, "c": -0.35},
    {"m": 0.35, "t": 0.01, "c": -0.4},
    {"m": 0.35, "t": 0.01, "c": -0.45},
    {"m": 0.35, "t": 0.01, "c": -0.5}   
]

for parameters_relevant_total_aquaint_item in parameters_relevant_total_aquaint_test_set_c:
    
    pivots_expo_total_unjudged_aquaint_item, minscore, maxscore = generate_pivots(qrels_aquaint, queries_aquaint, docids_aquaint, parameters_relevant_total_aquaint_item, params_unjudged_aquaint_2sys, 1000, True, "map", method = "exponential_unjudged", bin_samples = 10, bin_size = 0.1, max_runs = 10 )

    print("C")
    print(parameters_relevant_total_aquaint_item["c"])
    
    print("P10")
    p_10 = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "P_10")
    print(p_10)
    print("RR")
    rr = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "recip_rank")
    print(rr)
    print("MAP")
    map = get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "map")
    print(map)
    print("NDCG")
    ndcg =get_pivot_statistics(pivots_expo_total_unjudged_aquaint_item, queries_aquaint, qrels_aquaint, "ndcg")
    print(ndcg)


# In[ ]:





# In[43]:


pivots_expo_total_unjudged_aquaint_2sys_2gen, minscore, maxscore = generate_pivots(qrels_aquaint, queries_aquaint, docids_aquaint, params_total_relevant_aquaint_2sys, params_unjudged_aquaint_2sys, 1000, True, "map", method = "exponential_unjudged", bin_samples = 2, bin_size = 0.1, max_runs = 2 )


# In[71]:


map_aquaint_2sys_detailed = get_runs_statistics_detailed(two_systems_aquaint,  queries_aquaint, qrels_aquaint, "map")
p10_aquaint_2sys_detailed = get_runs_statistics_detailed(two_systems_aquaint, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint_2sys_detailed = get_runs_statistics_detailed(two_systems_aquaint, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint_2sys_detailed  = get_runs_statistics_detailed(two_systems_aquaint, queries_aquaint, qrels_aquaint, "ndcg")


print(map_aquaint_2sys_detailed)
print(p10_aquaint_2sys_detailed)
print(rr_aquaint_2sys_detailed)
print(ndcg_aquaint_2sys_detailed)


# In[65]:


map_aquaint_2sys_pivot_2gen = get_pivot_statistics_detailed(pivots_expo_total_unjudged_aquaint_2sys_2gen,  queries_aquaint, qrels_aquaint, "map")
p10_aquaint_2sys_pivot_2gen = get_pivot_statistics_detailed(pivots_expo_total_unjudged_aquaint_2sys_2gen, queries_aquaint, qrels_aquaint, "P_10")
rr_aquaint_2sys_pivot_2gen = get_pivot_statistics_detailed(pivots_expo_total_unjudged_aquaint_2sys_2gen, queries_aquaint, qrels_aquaint, "recip_rank")
ndcg_aquaint_2sys_pivot_2gen  = get_pivot_statistics_detailed(pivots_expo_total_unjudged_aquaint_2sys_2gen, queries_aquaint, qrels_aquaint, "ndcg")


print(map_aquaint_2sys_pivot_2gen)
print(p10_aquaint_2sys_pivot_2gen)
print(rr_aquaint_2sys_pivot_2gen)
print(ndcg_aquaint_2sys_pivot_2gen)


# In[ ]:




