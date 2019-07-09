#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import gensim
from gensim.models.fasttext import FastText as FT_gensim
import codecs


class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = '/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data/preprocessed_data/%s/train.txt' % domain
    model_file = '/mnt/cephfs/hadoop-compute/phoenix/arindam/projectKraken/data/unsupervised_aspect_data/preprocessed_data/%s/w2v_embedding' % domain
    sentences = Sentences(source)

    model = FT_gensim(size=200, window=3, min_count=100)  # instantiate
    # build the vocabulary
    model.build_vocab(corpus_file=sentences)

    # train the model
    model.train(
        corpus_file=sentences, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words
    )

    print(model)
    #model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4, sg=1, iter=5)
    model.save(model_file)


if __name__ == "__main__":
    print('Pre-training word embeddings ...')
    main('eaters')
