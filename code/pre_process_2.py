## preprocess
## Arindam  Arindam preprocess
## https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial/notebook


import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import io

import spacy  # For preprocessing

nlp = spacy.load('en_core_web_lg', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    print ("--",txt)
    if len(txt) > 2:
        return ' '.join(txt)

# f=io.open("/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data//datasets/eaters/train.txt", mode="r", encoding="utf-8")
# data_=f.read(1000)
# f.close()

lines = (line.rstrip('\n') for line in open("/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data//datasets/eaters/train.txt",encoding="utf-8"))

print("finished reading")

print ("***************")
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in lines)
print(brief_cleaning)
print ("*************")
t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
print (txt)
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

print (len(txt))

# process Unicode text
with io.open("/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data//preprocessed_data//eaters/train.txt", 'w', encoding='utf8') as f:
    for item in txt:
        f.write("%s\n" % item)
f.close()