#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import logging

import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class W2VEmbReader:

    def __init__(self, emb_path, emb_dim=None):

        logger.info('Loading embeddings from: ' + emb_path)
        self.emb_matrix = np.load(emb_path)
        # self.emb_matrix = np.delete(self.emb_matrix, (-1), axis=0)
        self.emb_dim = self.emb_matrix.shape[1]

        if emb_dim is not None:
            assert self.emb_dim == emb_dim

        self.vector_size = self.emb_matrix.shape[0]

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    def get_emb_matrix(self):

        # L2 normalization
        norm_emb_matrix = self.emb_matrix / np.linalg.norm(self.emb_matrix, axis=-1, keepdims=True)
        norm_emb_matrix[np.isnan(norm_emb_matrix)] = 1e-8
        return norm_emb_matrix

    def get_aspect_matrix(self, n_clusters):
        """
            We need it for initialization: KMeans-clustered word embeddings
        """

        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return norm_aspect_matrix.astype(np.float32)

    def get_emb_dim(self):
        return self.emb_dim
