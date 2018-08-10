#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import argparse
import codecs
import json

import numpy as np
from sklearn.metrics import classification_report

import utils as U

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True,
                    help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300,
                    help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50,
                    help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000,
                    help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14,
                    help="The number of aspects specified by users (default=14)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15,
                    help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20,
                    help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                    help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                    help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                    help="domain of the corpus {restaurant, beer}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1,
                    help="The weight of orthogonol regularizaiton (default=0.1)")

args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain
# out_dir = '../pre_trained_model/' + args.domain
U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}

from keras.preprocessing import sequence

###### Get test data #############

with open("../../data/finnish/prepared/word_idx_finnish.json") as fh:
    vocab = json.load(fh)
    vocab['<pad>'] = 0

dataset = np.load("../../data/finnish/prepared/dataset_finnish.npz")
test_x = dataset["test_X"]
test_x_pos = dataset["test_X_POS"]
overall_maxlen = test_x.shape[1]

dataset_annotated = np.load("../../data/finnish/prepared/dataset_finnish_annotated.npz")
test_y = dataset["train_y"]


############# Build model architecture, same as the model used for training #########
from model import create_model
import keras.backend as K
from optimizers import get_optimizer

optimizer = get_optimizer(args)


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)


model = create_model(args, overall_maxlen, vocab)

## Load the save model parameters
model.load_weights(out_dir + '/model_param')
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])


# Create a dictionary that map word index to word
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
att_weights, aspect_probs = test_fn([test_x, 0])

# Save attention weights on test sentences into a file
att_out = codecs.open(out_dir + '/att_weights', 'w', 'utf-8')
print('Saving attention weights on test sentences...')

for idx in range(len(test_y)):

    att_out.write('----------------------------------------\n')
    att_out.write(str(idx) + '\n')

    word_inds = [i for i in test_x[idx] if i != 0]
    line_len = len(word_inds)
    weights = att_weights[idx]
    weights = weights[(overall_maxlen - line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' ' + str(round(weights[j], 3)) + '\n')

    truths = [i for i in test_y[idx] if i != -1]
    assert len(words) == len(truths)
    assert len(words) == len(weights)



# #####################################################
# # Uncomment the below part for F scores
# #####################################################

# # cluster_map need to be specified manually according to the top words in each inferred aspect (save in aspect.log)
#
# # map for the pre-trained restaurant model (under pre_trained_model/restaurant)
# cluster_map = {
#     0: 'Food', 1: 'Miscellaneous', 2: 'Miscellaneous', 3: 'Food',
#     4: 'Miscellaneous', 5: 'Food', 6: 'Price',  7: 'Miscellaneous', 8: 'Staff',
#     9: 'Food', 10: 'Food', 11: 'Anecdotes', 12: 'Ambience', 13: 'Staff'
# }
#
#
# print('--- Results on %s domain ---' % (args.domain))
# test_labels = '../preprocessed_data/%s/test_label.txt' % (args.domain)
# prediction(test_labels, aspect_probs, cluster_map, domain=args.domain)
