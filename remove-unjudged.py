#!/usr/bin/python

import sys

qrels_filename = sys.argv[1]
run_filename = sys.argv[2]

qrels = []

with open(qrels_filename) as qrels_file:
    for line in qrels_file:
        topic, zero, document, evalution = line.split()
        topic_document = topic + " " + document
        qrels.append(topic_document)

unjudged_10 = 0
unjudged_100 = 0
unjudged_all = 0
newrank = 1

with open(run_filename) as run_file:
    for line in run_file:
        topic, zero, document, rank, score, description = line.split()
        description = description.rstrip("\n")
        topic_document = topic + " " + document

        if (rank == "1"):
            newrank = 1

        if topic_document in qrels:
            print(topic + " 0 " + document + " " + str(newrank) + " " + score + " " + description)
            newrank = newrank + 1
        else:
            if int(rank) <= 10:
                unjudged_10 = unjudged_10 + 1
            if int(rank) <= 100:
                unjudged_100 = unjudged_100 + 1
            unjudged_all = unjudged_all + 1

print (str(unjudged_10) + ", " + str(unjudged_100) + ", " + str(unjudged_all))
