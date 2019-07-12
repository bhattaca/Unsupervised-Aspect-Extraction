## preprocess
## Arindam  Arindam preprocess
## https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial/notebook


import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import io
import argparse
import utils as U

import spacy  # For preprocessing


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", dest="input_dir_path", type=str, metavar='<str>', required=True,
                    help="The path to the input directory", default="/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data//datasets/eaters/train.txt")

parser.add_argument("-o", "--out-dir", dest="output_dir_path", type=str, metavar='<str>', required=True,
                    help="The path to the output directory", default="/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data//preprocessed_data//eaters/train.txt")

args = parser.parse_args()
U.print_args(args)

#### spacy load model
nlp = spacy.load('en_core_web_lg', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
print(nlp.pipe_names)

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)


lines = (line.rstrip('\n') for line in open(args.input_dir_path, encoding="utf-8"))
lines = list(filter(None, lines)) # fastest

print("finished reading ")
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in lines)

t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

lines = txt.split('\n')
txt = [line + "\n" for line in lines if line.strip()]


print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

print (len(txt))

# process Unicode text
with io.open(args.output_dir_path, 'w', encoding='utf8') as f:
    for item in txt:
        f.write("%s\n" % item)
f.close()