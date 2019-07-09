#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import gensim
from gensim.models import FastText
import codecs


class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = '/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data/preprocessed_data/%s/train.txt' % domain
    ## using the original text as source
    source = '/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data/datasets/%s/train.txt" % domain
    model_file = '/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data/preprocessed_data/%s/w2v_embedding.bin' % domain
    sentences = Sentences(source)

    model = FastText(size=200, window=3, min_count=100)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    print("MODEL INFO : ",model)
    #model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4, sg=1, iter=5)
    model.save(model_file)


if __name__ == "__main__":
    print('Pre-training word embeddings ...')
    main('eaters')
