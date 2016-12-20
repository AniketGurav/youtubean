#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import time
import sys
import subprocess
import os
import random
import json
#import pastalog

from enlp.rnn.theano_elman_rnn import ElmanRNN
from enlp.rnn.theano_jordan_rnn import JordanRNN
from enlp.rnn.theano_lstm import LSTM
from enlp.rnn.theano_elman_brnn import ElmanBRNN
from enlp.rnn.theano_blstm import BLSTM

from enlp.rnn.utils import contextwin, minibatch, shuffle
from enlp.conll import conlleval

rnns = {"ElmanRNN": ElmanRNN,
        "JordanRNN": JordanRNN,
        "LSTM": LSTM,
        "ElmanBRNN": ElmanBRNN,
        "BLSTM": BLSTM}

# parse parameters
desc = "Help for train.py, a script that trains RNNs for opinion mining given a JSON file"

parser = argparse.ArgumentParser(description=desc)

parser.add_argument('json_path',
                    help="Path to the JSON file")

parser.add_argument('model_path',
                    help="Path to store model results")

parser.add_argument('rnn_type',
                    choices=rnns,
                    help="Type of RNN to use")

parser.add_argument('ne',
                    type=int,
                    help="Number of epochs")

parser.add_argument('ws',
                    type=int,
                    help="Word window context size")

parser.add_argument('nh',
                    type=int,
                    help="Size of hidden state")

parser.add_argument('bs',
                    type=int,
                    help="Batch size/ truncated-BPTT steps")

parser.add_argument('lr',
                    type=float,
                    help="Learning rate")

parser.add_argument('seed',
                    type=int,
                    help="Random seed")

parser.add_argument("--fine_tuning", "-ft",
                    action='store_true',
                    help="Add fine tuning")

parser.add_argument("--feat", "-f",
                    action='store_true',
                    help="Use linguistic features")

parser.add_argument("--decay", "-d",
                    action='store_true',
                    help="Use decay when no improvement")

parser.add_argument("--sentence_train", "-st",
                    action='store_true',
                    help="Train using sentence NLL (otherwise use batches)")

parser.add_argument("--verbose", "-v",
                    action='store_true',
                    help="Show training progress and results")

parser.add_argument("--logfile", "-l",
                    default='log.txt',
                    help="Logfile name. Default is log.txt")

parser.add_argument("--pastalog", "-p",
                    action='store_true',
                    help="send data to pastalog")

parser.add_argument("--weight", "-w",
                    default=1.0,
                    type=float,
                    help="Weight for domain adaptation")

args = parser.parse_args()

json_path = args.json_path
model_path = args.model_path
rnn_type = args.rnn_type
ne = args.ne
ws = args.ws
nh = args.nh
bs = args.bs
lr = args.lr
seed = args.seed
ft = args.fine_tuning
feat = args.feat
decay = args.decay
st = args.sentence_train
v = args.verbose
l = args.logfile
pl = args.pastalog
w = args.weight

folds = False

if os.path.isdir(json_path):
    json_file_paths = [os.path.join(json_path, name)
                       for name in sorted(os.listdir(json_path))]
    folds = True

else:
    json_file_paths = [json_path]

# containers for averaging k folds
best_f1s_test, best_f1s_valid = [], []


for counter, json_file_path in enumerate(json_file_paths):

    has_test, has_valid, has_domain = False, False, False

    # load the dataset
    corpus = json.load(open(json_file_path, "rb"))

    word2idx = corpus["word2idx"]
    label2idx = corpus["label2idx"]

    # check if corpus has validation/test data
    # which are not included when training for PRED domain adaptation
    if 'valid_x' in corpus:
        has_valid = True

    if 'test_x' in corpus:
        has_test = True

    if 'domain' in corpus:
        has_domain = True

    idx2label = dict((k, v) for v, k in label2idx.iteritems())
    idx2word = dict((k, v) for v, k in word2idx.iteritems())

    train_x, train_y = corpus['train_x'], corpus['train_y']
    train_x_feat = [np.asarray(m, dtype=np.float32)
                    for m in corpus['train_feat_x']]

    if has_domain and w < 1.0:
        train_weights = np.asarray(corpus['train_weights'], dtype=np.float32)
        train_weights = train_weights * w + (1 - train_weights)
    else:
        train_weights = np.ones((len(train_x,)), dtype=np.float32)

    if has_valid:
        valid_x, valid_y = corpus['valid_x'], corpus['valid_y']
        valid_x_feat = [np.asarray(m, dtype=np.float32)
                        for m in corpus['valid_feat_x']]
        if has_domain and w < 1.0:
            valid_weights = np.asarray(corpus['valid_weights'], dtype=np.float32)
            valid_weights = valid_weights * w + (1 - valid_weights)
        else:
            valid_weights = np.ones((len(valid_x, )), dtype=np.float32)

    if has_test:
        test_x, test_y = corpus['test_x'], corpus['test_y']
        test_x_feat = [np.asarray(m, dtype=np.float32)
                       for m in corpus['test_feat_x']]
        if has_domain and w < 1.0:
            test_weights = np.asarray(corpus['test_weights'], dtype=np.float32)
            test_weights = test_weights * w + (1 - test_weights)
        else:
            test_weights = np.ones((len(test_x, )), dtype=np.float32)

    embeddings = np.asarray(corpus['embeddings'], dtype=np.float32)

    if has_valid and has_test:
        # vocabulary size
        # vs = len(set(reduce(lambda x, y: list(x) + list(y), train_x + valid_x + test_x)))

        # number of classes

        nc = len(set(reduce(lambda x, y: list(x) + list(y), train_y + valid_y + test_y)))

    else:
        vs = len(set(reduce(lambda x, y: list(x) + list(y), train_x)))
        nc = len(set(reduce(lambda x, y: list(x) + list(y), train_y)))


    # number of sentences
    ns = len(train_x)

    # feat dim
    if feat:
        fd = corpus["featdim"]
    else:
        fd = 0

    # instantiate the model
    np.random.seed(seed)
    random.seed(seed)

    # init model

    RNN = rnns.get(rnn_type, None)
    if RNN is None:
        raise Exception("Wrong RNN type provided.")

    rnn = RNN(nh, nc, ws, embeddings, featdim=fd,
              fine_tuning=ft, truncate_gradient=bs)

    # TRAINING (with early stopping on validation set)
    best_f1_test, best_f1_valid = -np.inf, -np.inf
    best_p_test, best_p_valid = -np.inf, -np.inf
    best_r_test, best_r_valid = -np.inf, -np.inf
    best_epoch = 0

    # create a folder to store the models
    if folds:
        base_path, fold_name = os.path.split(json_file_path)
        fold_name = fold_name.replace(".json", "")
        dataset_name = os.path.basename(base_path)
        dataset_model_path = os.path.join(model_path, dataset_name, fold_name)

    else:
        dataset_name = os.path.basename(json_file_path).replace(".json", "")
        dataset_model_path = os.path.join(model_path, dataset_name)

    if not os.path.exists(dataset_model_path):
        os.makedirs(dataset_model_path)

    omit = ["model_path", "json_path", "verbose", "logfile", "pastalog"]
    params_dict = vars(args)
    params = " ".join([key + "=" + str(value)
                       for key, value in sorted(params_dict.items())
                       if key not in omit])

    params_dataset_model_path = os.path.join(dataset_model_path, params)

    if not os.path.exists(params_dataset_model_path):
        os.makedirs(params_dataset_model_path)

    # start pastalog (default address)
    if pl:
        plog = pastalog.Log("http://localhost:8120",
                            dataset_name + " " + params)

    for e in xrange(ne):
        shuffle([train_x, train_x_feat, train_y, train_weights], seed)
        tic = time.time()
        nlls = []
        for i in xrange(len(train_x)):
            cwords = contextwin(train_x[i], ws)
            labels = train_y[i]
            weight = train_weights[i]
            if st:
                features = train_x_feat[i]
                words = map(lambda x: np.asarray(x).astype('int32'), cwords)
                sentence_nll = rnn.sentence_train(words, features, labels, lr, weight)
                nlls.append(sentence_nll)
                if ft:
                    rnn.normalize()

            else:
                words = map(lambda x: np.asarray(x).astype('int32'),
                            minibatch(cwords, bs))
                features = minibatch(train_x_feat[i], bs)
                bnlls = []
                for word_batch, feat_batch, label_last_word in zip(words, features, labels):
                    nll = rnn.train(word_batch, feat_batch, label_last_word, lr, weight)
                    bnlls.append(nll)
                    if ft:
                        rnn.normalize()
                nlls.append(np.mean(bnlls))

            if v:
                print '[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / ns),\
                       'completed in %.2f (sec) <<\r' % (time.time() - tic),
                sys.stdout.flush()

        if has_valid:
            predictions_valid = [map(lambda x: idx2label[x],
                                     rnn.classify(np.asarray(contextwin(x, ws)).astype('int32'), feat_x, valid_weight))
                                 for x, feat_x, valid_weight in zip(valid_x, valid_x_feat, valid_weights)]
            ground_truth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
            words_valid = [map(lambda x: idx2word[x], w) for w in valid_x]
            res_valid = conlleval(predictions_valid, ground_truth_valid,
                                  words_valid, params_dataset_model_path + '/current.valid.txt')

        if has_test:
            predictions_test = [map(lambda x: idx2label[x],
                                    rnn.classify(np.asarray(contextwin(x, ws)).astype('int32'), feat_x, test_weight))
                                for x, feat_x, test_weight in zip(test_x, test_x_feat, test_weights)]
            ground_truth_test = [map(lambda x: idx2label[x], y) for y in test_y]
            words_test = [map(lambda x: idx2word[x], w) for w in test_x]
            res_test = conlleval(predictions_test, ground_truth_test,
                                 words_test, params_dataset_model_path + '/current.test.txt')

        # do stuff when provided valid and test data
        if has_valid and has_test:
            if pl:
                plog.post("train nll", float(np.mean(nlls)), e)
                plog.post("valid f1", float(res_valid['f1']), e)
                plog.post("test f1", float(res_test['f1']), e)

            if res_test['f1'] > best_f1_test:
                rnn.save(params_dataset_model_path)
                best_f1_test, best_f1_valid = res_test['f1'], res_valid['f1']
                best_p_test, best_p_valid = res_test['p'], res_valid['p']
                best_r_test, best_r_valid = res_test['r'], res_valid['r']
                best_epoch = e

                if v:
                    print 'NEW BEST: epoch', e, 'valid F1', best_f1_valid, 'best test F1', best_f1_test, ' '*20

                subprocess.call(['mv', params_dataset_model_path + '/current.test.txt',
                                 params_dataset_model_path + '/best.test.txt'])
                subprocess.call(['mv', params_dataset_model_path + '/current.valid.txt',
                                 params_dataset_model_path + '/best.valid.txt'])
            else:
                print ''

            # learning rate decay if no improvement in 10 epochs
            if decay and abs(best_epoch-e) >= 10:
                lr *= 0.5

            if lr < 1e-5:
                break

        # if no valid/test data provided, just save the model every epoch
        else:
            rnn.save(params_dataset_model_path)

    if has_valid and has_test:
        best_f1s_test.append(best_f1_test)
        best_f1s_valid.append(best_f1_valid)

        logfile = os.path.join(model_path, l)

        if folds:
            # log result on fold
            with open(logfile, 'a') as log:
                log.write("Dataset=" + dataset_name +
                          "\t" + params +
                          "\tFold=" + str(counter + 1) +
                          "\tBest F1=" + str(best_f1_test) +
                          "\tBest R=" + str(best_r_test) +
                          "\tBest P=" + str(best_p_test) + "\n")
        else:
            with open(logfile, 'a') as log:
                log.write("Dataset=" + dataset_name +
                          "\t" + params +
                          "\tBest F1=" + str(best_f1_test) +
                          "\tBest R=" + str(best_r_test) +
                          "\tBest P=" + str(best_p_test) + "\n")

if folds and has_valid and has_test:
    # print average for folds
    folds = len(json_file_paths)
    print 'FOLDS AVG BEST: valid F1', 1.0*sum(best_f1s_valid)/folds,\
          'best test F1', 1.0*sum(best_f1s_test)/folds, ' ' * 20
