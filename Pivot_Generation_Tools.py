#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


pwd()


# In[3]:


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


# In[4]:


def get_performance_detailed (run, queries, qrels, measure):
    #print(run, queries, qrels)
    
    e = pt.Experiment(
        [run],
        queries,
        qrels,
        eval_metrics=[measure],
        perquery=True

    )
    
    #print("hi world")
    
    #print(e['value'])
    return e['value']


# In[5]:


# Calculate mean and the STD of the pivots

import statistics 

def get_runs_statistics(runs, queries, qrels, measure):
    
    scores = []
    
    i = 0
    while i < len(runs):
        
        run = runs[i]
        score = get_performance(run, queries, qrels, measure)
        #print(score)

        scores.append(score)            

        i = i + 1
        
    return(statistics.mean(scores), statistics.stdev(scores))


# In[6]:


# Calculate mean and the STD of the pivots

import statistics 

def get_runs_statistics_detailed(runs, queries, qrels, measure):
    
    scores_all = []
    
    i = 0
    while i < len(runs):
        
        run = runs[i]
        scores = get_performance_detailed(run, queries, qrels, measure)
        #print(score)

        scores_all.extend(scores)            

        i = i + 1
        
    return(statistics.mean(scores_all), statistics.stdev(scores_all))


# In[7]:


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


# In[8]:


def get_pivot_statistics_detailed(pivots, queries, qrels, measure):
    
    scores_all = []
    
    print("here")
    
    print(len(pivots))
    
    i = 0
    while i <= 1000:
        
        if (i in pivots):
        
            b = 0

            while b < len(pivots[i]):

                pivot = pivots[i][b]
                scores = get_performance_detailed(pivot, queries, qrels, measure)
                print(scores)

                scores_all.extend(scores)

                b = b+1        

        i = i + 1
        
    return(statistics.mean(scores_all), statistics.stdev(scores_all))


# In[9]:


def monoExp(x, m, t, c):
    return m * np.exp(-t * x) + c


# In[2]:


# Print out the statistics of the relevant and irrelevant documents 

def print_runs_distributions(runs, queries, qrels):
       
    outrows = []
    i = 0
   
    print(len(runs))
    print(runs[0])
    
    print("GD0")
    
    while i < len(runs):
                          
        print(i)
        
        run = runs[i]
                    
        for query_id in sorted(set(run['qid'])):

            print (query_id)

            query_run = run[run['qid'] == query_id]
            qrels_run = qrels[qrels['qid'] == query_id]

            r = 0

            for doc_id in query_run['docno']:

                relevance = "U"
                            
                # HERE
                #if (doc_id == -1):
                #    print ("X")
                #    relevance = "X"

                if (not (qrels_run[qrels_run['docno'] == doc_id]).empty):
                    qrel_line = qrels_run[qrels_run['docno'] == doc_id]
                    #print(qrel_line)

                    if (qrel_line.values[0][2] > 0):
                        relevance = "R"
                    else:
                        relevance = "N"

                #print(r)
                if (r < len(outrows)):
                    old_row = outrows[r]
                    new_row = old_row + relevance
                    outrows[r] = new_row
                    #print("append")

                else:
                    outrows.append(relevance)
                    #print ("newrow")
                    #print (relevance, end='')
                        
                r = r + 1
                        
        i = i + 1
        
    relevant = []
    irrelevant = []
    unjudged = []
    not_enough = []
    
    for row in outrows:
        relevanti = row.count("R")
        irrelevanti = row.count("N")
        unjudgedi = row.count("U")
        not_enoughi = row.count("X")
        
        relevant.append(relevanti)
        irrelevant.append(irrelevanti)
        unjudged.append(unjudgedi)
        not_enough.append(not_enoughi)
        
    #print(outrows)
    
    print(relevant, irrelevant, unjudged, not_enough)
    
    return(relevant, irrelevant, unjudged, not_enough)


# In[3]:


def print_pivots_distributions(runs, queries, qrels):

       
    outrows = []
    i = 0
   
    print(len(runs))
    print(runs)
    
    while i <= 1000:
               
        if (i in runs):
            
            print(i)
        
            if (type(runs[i]) is list):

                b = 0
                
                while b < len(runs[i]):
                    
                    print(b)

                    run = runs[i][b]
                    
                    for query_id in sorted(set(run['qid'])):

                        print (query_id)

                        query_run = run[run['qid'] == query_id]
                        qrels_run = qrels[qrels['qid'] == query_id]

                        r = 0

                        for doc_id in query_run['docno']:

                            relevance = "U"

                            # HERE
                            #if (doc_id == -1):
                            #    print ("X")
                            #    relevance = "X"

                            if (not (qrels_run[qrels_run['docno'] == doc_id]).empty):
                                qrel_line = qrels_run[qrels_run['docno'] == doc_id]
                                #print(qrel_line)

                                if (qrel_line.values[0][2] > 0):
                                    relevance = "R"
                                else:
                                    relevance = "N"

                            #print(r)
                            if (r < len(outrows)):
                                old_row = outrows[r]
                                new_row = old_row + relevance
                                outrows[r] = new_row
                                #print("append")

                            else:
                                outrows.append(relevance)
                                #print ("newrow")
                                #print (relevance, end='')

                            r = r + 1

                    b = b+1

            else:
                run = runs[i]

                #print("here")        

                for query_id in sorted(set(run['qid'])):

                    #print (query_id)

                    query_run = run[run['qid'] == query_id]
                    qrels_run = qrels[qrels['qid'] == query_id]

                    r = 0

                    for doc_id in query_run['docno']:

                        relevance = "U"

                        if (not (qrels_run[qrels_run['docno'] == doc_id]).empty):
                            qrel_line = qrels_run[qrels_run['docno'] == doc_id]
                            #print(qrel_line)

                            if (qrel_line.values[0][2] > 0):
                                relevance = "R"
                            else:
                                relevance = "N"

                        #print(r)
                        if (r < len(outrows)):
                            old_row = outrows[r]
                            new_row = old_row + relevance
                            outrows[r] = new_row
                            #print("append")

                        else:
                            outrows.append(relevance)
                            #print ("newrow")

                        #print (relevance, end='')

                        r = r + 1
                        
        i = i + 1
        
    relevant = []
    irrelevant = []
    unjudged = []
    not_enough = []
    
    for row in outrows:
        relevanti = row.count("R")
        irrelevanti = row.count("N")
        unjudgedi = row.count("U")
        not_enoughi = row.count("X")
        
        relevant.append(relevanti)
        irrelevant.append(irrelevanti)
        unjudged.append(unjudgedi)
        not_enough.append(not_enoughi)
        
    #print(outrows)
    
    print(relevant, irrelevant, unjudged, not_enough)
    
    return(relevant, irrelevant, unjudged, not_enough)
    
    #return(relevant, irrelevant, unjudged, not_enough, n)


# In[12]:


# Get counts, statistics and parameters for the collection

def get_distributions(qrels, queries, documents, runs, query_ids, run_type):

    # Filtering might be done on the query_ids 
    if query_ids:
       
        newruns = []

        for queryid in query_ids:
            
            newrun = []
            for run in runs:
                nr = run[run["qid"] == str(queryid)]
                newrun.append(nr)
           
            newruns.append(newrun)
                
        runs = newruns

    if run_type == "run":
        relevant, irrelevant, unjudged, missing = print_runs_distributions(runs, queries, qrels)
    if run_type == "pivot":
        relevant, irrelevant, unjudged, missing = print_pivots_distributions(runs, queries, qrels)
    
    print ("GD2")
    print(relevant, irrelevant, unjudged)

    relevant_array = np.array(relevant, dtype=np.int)
    irrelevant_array = np.array(irrelevant, dtype=np.int)
    unjudged_array = np.array(unjudged, dtype=np.int)
    
    positions = range(1,1001,1)

    relevant_proportion = []
    relevant_total_proportion = []
    unjudged_proportion = []

    
    print ("GD3")

    i = 0
    while i < len(relevant_array):
        num1 = relevant_array[i]
        num2 = irrelevant_array[i]
        num3 = unjudged_array[i]
        rp = 0
        if (not (num1+num2 == 0)):
            rp = (num1 / (num1 + num2))
        print (i, num1, num2, num3, rp)
        relevant_proportion.append(rp)
        relevant_total_proportion.append(num1 / (num1 + num2 + num3))
        unjudged_proportion.append(num3 / (num1 + num2 + num3))
        i = i+1
    
    init = (2000, .1, 50) # start with values near those we expect
    
    print ("GD4")

    print(relevant_proportion, relevant_total_proportion, unjudged_proportion)
    
    params_relevant_array, cv = sp.optimize.curve_fit(monoExp, positions, relevant_proportion, init)
    params_relevant = {"m": params_relevant_array[0], "t": params_relevant_array[1], "c": params_relevant_array[2]}
    
    params_relevant_total_array, cv = sp.optimize.curve_fit(monoExp, positions, relevant_total_proportion, init)
    params_relevant_total = {"m": params_relevant_total_array[0], "t": params_relevant_total_array[1], "c": params_relevant_total_array[2]}
    
    params_unjudged_array, cv = sp.optimize.curve_fit(monoExp, positions, unjudged_proportion, init)
    params_unjudged = {"m": params_unjudged_array[0], "t": params_unjudged_array[1], "c": params_unjudged_array[2]}
    
    return(params_relevant, params_relevant_total, params_unjudged)


# In[4]:


def generate_expo_run (qrels, queries, documents, method, runlen, randord, relevant_parameters, unjudged_parameters, p=0.5, seednum=1):
    
    #np.random.seed(seednum)
    
    print(runlen, randord)
    
    run = []

    for query_id in queries['qid']:
        
        judged_documents_per_query = qrels[qrels['qid'] == query_id]
    
        qrels_relevant_per_query = judged_documents_per_query[judged_documents_per_query["label"] > 0]['docno']
        qrels_irrelevant_per_query = judged_documents_per_query[judged_documents_per_query["label"] == 0]['docno']
        
        qrels_relevant_per_query_array = np.array(qrels_relevant_per_query)
        qrels_irrelevant_per_query_array = np.array(qrels_irrelevant_per_query)
        
        if randord == True:
            np.random.shuffle(qrels_relevant_per_query_array)
            np.random.shuffle(qrels_irrelevant_per_query_array)
            np.random.shuffle(documents)
        
        i = 0
        relevant_i = 0 
        irrelevant_i = 0
        unjudged_i = 0
        
        while (i < runlen):
        
            # Generate relevant or irrelevant?
            r = random.random()
            
            document = "-1"
            
            #print(str(r), str(p*10))
            
            t_parameter = relevant_parameters["t"]
            
            if "distort" in method:
                #beta = np.random.beta(1.4662276058413575, 9058718.350828953)
                
                distortion_parameter = 0.5 + random.random()
                
                t_parameter = relevant_parameters["t"] * distortion_parameter
                #print(str(relevant_parameters["t"]), str(t_parameter))            
            
            limit = monoExp(i,relevant_parameters["m"],t_parameter,relevant_parameters["c"])
            
            if (r <= limit):
                
                if (relevant_i < len(qrels_relevant_per_query_array)):
                    document = qrels_relevant_per_query_array[relevant_i]
                    print("R", end='')
                else:
                    print("X", end='')
                 
                relevant_i = relevant_i + 1 
            else:
                
                if ("unjudged" in method):
                    limit_unjudged = monoExp(i, unjudged_parameters["m"], unjudged_parameters["t"], unjudged_parameters["c"])
                    #limit_unjudged = root_squared(i, unjudged_parameters["a"], unjudged_parameters["b"])
                    #print(i)
                    #print(limit_unjudged)
                    
                    limit_relevant_unjudged = limit + limit_unjudged
                    
                    #print (limit_relevant_unjudged)
                    
                    if (r <= limit_relevant_unjudged):

                        if (unjudged_i < len(documents)):
                            document = documents[unjudged_i]
                            
                            while ((document in qrels_relevant_per_query_array or document in qrels_irrelevant_per_query_array) and (unjudged_i < len(documents))):
                                unjudged_i = unjudged_i + 1
                                document = documents[unjudged_i]

                            #print(document)
                            print("U", end='')
                            unjudged_i = unjudged_i + 1
                            
                    else:
                        if (irrelevant_i < len(qrels_irrelevant_per_query_array)):
                            document = qrels_irrelevant_per_query_array[irrelevant_i]

                            print("N", end='')                
                        else:
                            print("X", end='')
                            
                        irrelevant_i = irrelevant_i + 1 
                                
                else:             
                    if (irrelevant_i < len(qrels_irrelevant_per_query_array)):
                        document = qrels_irrelevant_per_query_array[irrelevant_i]
                        print("N", end='')                
                    else:
                        print("X", end='')
                    irrelevant_i = irrelevant_i + 1 
               
            row_dict = {}
            #print (document)
            row_dict["qid"] = query_id
            row_dict["docid"] = document
            row_dict["docno"] = document
            row_dict["rank"] = i
            row_dict["score"] = 1000 - i
            row_dict["query"] = "pivot_" + str(seednum)
            run.append(row_dict)
            #print(run)
            i = i + 1
            
        print("")

    run_df = pd.DataFrame(run)
    return(run_df)    


# In[14]:


def generate_pivots (qrels, queries, documents, relevant_parameters, unjudged_parameters, runlen, randord, measure = "map", method = "percentage", bin_samples = 10, bin_size = 0.1, max_runs = 10 ):

    i = 0
    min_measure = 1000
    max_measure = 0
    
    # Bins might need to be changed according to the measure, right now we assume, that the measure returns the scores between 0 and 1
    bins = {}
    
    if ("exponential" in method):
        
        b = 0
        
        while b < bin_samples:
            
            print(b)
            print(runlen, randord)
                       
            runi = generate_expo_run(qrels, queries, documents, method, runlen, randord, relevant_parameters, unjudged_parameters, seednum=int(b))
            scorei = pt.Utils.evaluate(runi, qrels, metrics=[measure], perquery=False)[measure]
            print(scorei)
                
            if min_measure > scorei:
                min_measure = scorei
            if max_measure < scorei:
                max_measure = scorei

            bin = math.floor(scorei*10)

            if bin in bins:
                bins[bin].append(runi)
                print('appending')
            else:
                bins[bin] = [runi]     
                
            b = b + 1
    
    if (method == "percentage"):
        
        while i < 1:
            
            b = 0
            
            while b < bin_samples:
            
                runi = generate_run(qrels, queries, runlen, randord, p=i, seednum=int(i*b*10))
                scorei = pt.Utils.evaluate(runi, qrels, metrics=[measure], perquery=False)[measure]
                
                all = pt.Utils.evaluate(runi, qrels, metrics=[measure], perquery=True)
                print(all)
                
                print (scorei)

                if min_measure > scorei:
                    min_measure = scorei
                if max_measure < scorei:
                    max_measure = scorei

                bin = math.floor(scorei*10)

                if bin in bins:
                    bins[bin].append(runi)
                    print('appending')
                else:
                    bins[bin] = [runi]     
                    
                b = b + 1
            
            i = i + 0.1
    
    if (method == "stratified"):
        while i < max_runs:

            # Right now, change the seeds one by i -- this should be ranodmized later
            
            seednum = i

            runi = generate_run_random(seednum, qrels, queries)
            scorei = pt.Utils.evaluate(runi, qrels, metrics=[measure], perquery=False)[measure]
            #print (scorei)

            if min_measure > scorei:
                min_measure = scorei
            if max_measure < scorei:
                max_measure = scorei

            bin = math.floor(scorei/bin_size)
            #print (bin)

            if bin in bins:
                bin_len = len(bins[bin])
                if (bin_len < bin_samples):
                    bins[bin].append(runi)
            else:
                bins[bin] = [runi]

            i = i + 1
        
    return (bins, min_measure, max_measure)
    


# In[15]:


def get_maximum_score (qrels, queries, measure):
    
    run = []
    
    for query_id in queries['qid']:
        
        judged_documents_per_query = qrels[qrels['qid'] == query_id]
    
        qrels_relevant_per_query = judged_documents_per_query[judged_documents_per_query["label"] > 0]['docno']       
        qrels_relevant_per_query_array = np.array(qrels_relevant_per_query)       
        
        i = 0
        relevant_i = 0 
        
        while (i < 1000):
        
            document = "-1"
        
            if (relevant_i < len(qrels_relevant_per_query_array)):
                document = qrels_relevant_per_query_array[relevant_i]
                
            row_dict = {}
            #print (document)
            row_dict["qid"] = query_id
            row_dict["docid"] = document
            row_dict["docno"] = document
            row_dict["rank"] = i
            row_dict["score"] = 1000 - i
            row_dict["query"] = "pivot_max"
            run.append(row_dict)
            #print(run)
            i = i + 1
            relevant_i = relevant_i + 1

    run_df = pd.DataFrame(run)
    
    score = pt.Utils.evaluate(run_df, qrels, metrics=[measure], perquery=False)[measure]
    
    return(score)    


# In[ ]:




