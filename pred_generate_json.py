#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import re
from collections import defaultdict
import os
import random
import warnings

from enlp.corpus.youtube import SamsungGalaxyS5
from enlp.corpus.semeval import SemEval14LaptopsTrain, SemEval14RestaurantsTrain
from enlp.embeddings import GoogleNews, SennaEmbeddings, WikiDeps

from enlp.rnn.theano_elman_rnn import ElmanRNN
from enlp.rnn.theano_jordan_rnn import JordanRNN
from enlp.rnn.theano_lstm import LSTM
from enlp.rnn.theano_elman_brnn import ElmanBRNN
from enlp.rnn.theano_blstm import BLSTM
from enlp.rnn.utils import contextwin


rnns = {"ElmanRNN": ElmanRNN,
        "JordanRNN": JordanRNN,
        "LSTM": LSTM,
        "ElmanBRNN": ElmanBRNN,
        "BLSTM": BLSTM}


def process_token(token):
    """Preprocessing for each token.

    As Liu et al. 2015 we lowercase all words
    and replace numbers with 'DIGIT'.

    Args:
        token (Token): Token

    Returns:
        str: processed token
    """
    ptoken = token.string.lower()
    if re.match("\\d+", ptoken):
        ptoken = "".join("DIGIT" for char in ptoken)
    return ptoken


def sent_to_aspect_iob(sentence, sentiment=True):
    """Builds a list of aspect IOB tags based on the
    provided sentence.

    Args:
        sentence (Sentence): Sentence object
        sentiment (bool): True if add sentiment (-1, 0 or 1) to the IOB label

    Returns:
        list: list of IOB tags
    """
    if not sentence.is_tokenized:
        raise Exception("Sentence not tokenized")
    tags = ["O"] * len(sentence)
    for aspect in sentence.aspects:
        orientation = int(aspect.orientation)
        if orientation > 0:
            orientation = "+"
        elif orientation == 0:
            orientation = "O"
        elif orientation < 0:
            orientation = "-"
        for position in aspect.positions:
            # only add aspects with known position
            if position:
                a_start, a_end = position
                start = None
                end = None
                for token in sentence:
                    if token.start <= a_start < token.end:
                        start = token.index
                for token in sentence:
                    if token.start < a_end <= token.end:
                        end = token.index
                if None not in [start, end]:
                    if start == end:
                        tags[start] = "B-TARG"
                        if sentiment:
                            tags[start] += orientation
                    elif end - start >= 1:
                        tags[start] = "B-TARG"
                        if sentiment:
                            tags[start] += orientation
                        for i in range(start + 1, end + 1):
                            tags[i] = "I-TARG"
                            if sentiment:
                                tags[i] += orientation
    assert len(tags) == len(sentence)
    return tags


def sent_to_bin_feats(sentence, funcs):
    """Generates binary one-hot features from a sentence.
    Applies each function in funcs to each token
    in the provided sentence.

    Args:
        sentence (Sentence):
            Sentence object
        funcs (list):
            list of functions to apply to
            each token in the sentence

    Returns:
        numpy.array : dim=(len(sentence), len(feat_funcs)
    """
    if not sentence.is_tokenized:
        raise Exception("Sentence not tokenized")
    matrix = np.zeros((len(sentence), len(funcs)))
    for i, token in enumerate(sentence):
        for j, func in enumerate(funcs):
            if func(token):
                matrix[i, j] = 1
    return matrix


def to_one_hot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result


def build_json_pred_source(from_corpus, to_corpus, embeddings, json_path,
                           feat_funcs=[], sentiment=True):
    """
    Generate a JSON file for training a model on the complete SRC dataset (no valid or test data splitting
    is included) for the PRED domain adaptation technique.

    The generated JSON also contains the information regarding vocabulary, embeddings and
    data on the TGT domain to be used to generate the new JSON.

    Args:
        from_corpus (Corpus):
            SRC Corpus object
        to_corpus (Corpus)
            TGT Corpus object
        embeddings (Embeddings):
            Embeddings object
        feat_funcs (list):
            list of functions to extract binary features from tokens, None if no
            bynary features should be extracted (default=None)
        ratio (float):
            Development/Test ratio (default=0.8)
        sentiment (bool):
            True if add sentiment (-1, 0 or 1) label to the aspect

    Returns:
        name (str) of the processed corpus if successful,
        or None if it fails
    """

    if sentiment:
        label2idx = {"O": 0,
                     "B-TARG+": 1,
                     "I-TARG+": 2,
                     "B-TARG-": 3,
                     "I-TARG-": 4,
                     "B-TARGO": 5,
                     "I-TARGO": 6}
    else:
        label2idx = {"O": 0, "B-TARG": 1, "I-TARG": 2}

    counts = defaultdict(int)
    word2idx = dict()
    aspect_words = []
    from_psentences, from_plabels, from_features = [], [], []
    to_psentences, to_plabels, to_features, = [], [], []

    # process from_corpus
    for sentence in from_corpus.sentences:
        psentence = []
        for word in sentence.tokens:
            pword = process_token(word)
            counts[pword] += 1
            psentence.append(pword)
        from_psentences.append(psentence)
        labels = sent_to_aspect_iob(sentence, sentiment=sentiment)
        from_plabels.append(labels)

        # collect words that are labeled as aspects
        for i, label in enumerate(labels):
            if label != "O":
                aspect_words.append(sentence[i].string)

        if feat_funcs:
            from_features.append(sent_to_bin_feats(sentence, feat_funcs))

    # process to_corpus
    for sentence in to_corpus.sentences:
        psentence = []
        for word in sentence.tokens:
            pword = process_token(word)
            counts[pword] += 1
            psentence.append(pword)
        to_psentences.append(psentence)
        labels = sent_to_aspect_iob(sentence, sentiment=sentiment)
        to_plabels.append(labels)

        # collect words that are labeled as aspects
        for i, label in enumerate(labels):
            if label != "O":
                aspect_words.append(sentence[i].string)

        if feat_funcs:
            to_features.append(sent_to_bin_feats(sentence, feat_funcs))

    # get words with count == 1 and not labeled as aspects
    highcounts = {k: v for k, v in counts.iteritems()
                  if v > 1 and k in embeddings}

    # adding +2 for PADDING and UNKNOWN
    vocab_size = len(highcounts) + 2
    vector_size = embeddings.vector_size
    matrix = np.zeros((vocab_size, vector_size), dtype=np.float32)

    # first item is the unseen or unknown token
    matrix[0] = embeddings.unseen()

    # vector on index -1 corresponds to PADDING, as
    # used by contextwin function in in rnn.utils

    i = 1
    for word in counts:
        if word in highcounts:
            word2idx[word] = i
            matrix[i] = embeddings[word]
            i += 1
        else:
            if word not in embeddings:
                word2idx[word] = 0
            else:
                if counts[word] == 1:
                    word2idx[word] = 0
                else:
                    word2idx[word] = i
                    matrix[i] = embeddings[word]
                    i += 1

    assert i + 1 == vocab_size

    json_filename = "PRED" + \
                    "[" + ".".join(from_corpus.pipeline) + "]" + from_corpus.name + "." + \
                    "[" + ".".join(to_corpus.pipeline) + "]" + to_corpus.name + "." + \
                    embeddings.name + ".json"

    # add all the data in 'from_corpus' as training
    train_x, train_feat_x, train_y, = [], [], []
    for i in range(len(from_psentences)):
        train_x.append([word2idx[word] for word in from_psentences[i]])
        train_feat_x.append(from_features[i].tolist())
        train_y.append([label2idx[label] for label in from_plabels[i]])

    # add all the data in 'to_corpus' as extra
    extra_x, extra_feat_x, extra_y = [], [], []
    for i in range(len(to_psentences)):
        extra_x.append([word2idx[word] for word in to_psentences[i]])
        extra_feat_x.append(to_features[i].tolist())
        extra_y.append([label2idx[label] for label in to_plabels[i]])

    jsondic = dict()
    jsondic["featdim"] = len(feat_funcs)
    jsondic["train_x"], jsondic["train_feat_x"], jsondic["train_y"] = train_x, train_feat_x, train_y
    jsondic["extra_x"], jsondic["extra_feat_x"], jsondic["extra_y"] = extra_x, extra_feat_x, extra_y
    jsondic["word2idx"] = word2idx
    jsondic["label2idx"] = label2idx
    jsondic["embeddings"] = matrix.tolist()

    with open(os.path.join(json_path, json_filename), "wb") as json_file:
        try:
            json.dump(jsondic, json_file)
        except Exception as e:
            warnings.warn(str(e))
            return None
        finally:
            print "Written " + json_filename
            print ""
            return json_filename


def build_json_pred_target(json_path, model_path, folds=10):
    """
    Add PRED faetures generated used the trained model in model_path (trained on the SRC data)
    by FF on the TGT data, and generate a new json file for training on the TGT domain including
    these features. The name of the json_file used to train the model in model_path is obtained
    from the path.

    Args:
        json_path (str):
            path to the JSON files
        model_path (str):
            path to the model trained on SRC domain
        folds (int)
            number of folds to use for training on the TGT domain (default=10)

    Returns:
        name of the generated JSON file
    """

    basedir, params = os.path.split(model_path)
    _, json_filename = os.path.split(basedir)

    # load the trained model
    for rnn_name in rnns:
        if rnn_name in params:
            RNN = rnns[rnn_name]

    rnn = RNN.load(model_path)
    ws = rnn.hyperparams["cs"]
    nc = rnn.hyperparams["nc"]

    # load the dataset
    corpus = json.load(open(os.path.join(json_path, json_filename + ".json"), "rb"))

    extra_x, extra_y = corpus['extra_x'], corpus['extra_y']
    extra_x_feat = corpus['extra_feat_x']
    extra_x_feat = [np.asarray(feat_x, dtype=np.float32) for feat_x in extra_x_feat]

    domain_x_feat = [rnn.classify(np.asarray(contextwin(x, ws)).astype('int32'), feat_x)
                     for x, feat_x, in zip(extra_x, extra_x_feat)]

    # add domain features to x_feat
    extra_x_feat = [np.concatenate((a, to_one_hot(b, num_classes=nc)), axis=1)
                    for a, b in zip(extra_x_feat, domain_x_feat)]

    # calcule fold sizes
    foldsize = 1.0 * (len(extra_x)) / folds
    rest = 1.0 * (len(extra_x)) % folds

    json_path_name = json_filename

    json_path = os.path.join(json_path, json_path_name)
    if not os.path.isdir(json_path):
        os.makedirs(json_path)

    result = []

    for f in range(1, folds + 1):

        start = (f - 1) * foldsize
        end = f * foldsize

        train_size = foldsize * (folds - 1)

        if f == folds:
            end += rest

        train_ids, valid_ids, test_ids = [], [], []
        train_x, train_feat_x, train_y = [], [], []
        valid_x, valid_feat_x, valid_y = [], [], []
        test_x, test_feat_x, test_y = [], [], []

        for i in range(len(extra_x)):
            # test
            if start <= i < end:
                test_ids.append(i)
            # development
            else:
                rr = random.random()
                if rr < 0.9 and len(train_ids) < train_size:
                    train_ids.append(i)
                else:
                    valid_ids.append(i)

        for i in train_ids:
            train_x.append(extra_x[i])
            train_feat_x.append(extra_x_feat[i].tolist())
            train_y.append(extra_y[i])

        for i in valid_ids:
            valid_x.append(extra_x[i])
            valid_feat_x.append(extra_x_feat[i].tolist())
            valid_y.append(extra_y[i])

        for i in test_ids:
            test_x.append(extra_x[i])
            test_feat_x.append(extra_x_feat[i].tolist())
            test_y.append(extra_y[i])

        jsondic = dict()
        jsondic["featdim"] = int(corpus['featdim']) + nc
        jsondic["fold"] = f
        jsondic["train_x"], jsondic["train_feat_x"], jsondic["train_y"] = train_x, train_feat_x, train_y
        jsondic["valid_x"], jsondic["valid_feat_x"], jsondic["valid_y"] = valid_x, valid_feat_x, valid_y
        jsondic["test_x"], jsondic["test_feat_x"], jsondic["test_y"] = test_x, test_feat_x, test_y

        jsondic["word2idx"] = corpus["word2idx"]
        jsondic["label2idx"] = corpus["label2idx"]
        jsondic["embeddings"] = corpus["embeddings"]

        json_filename = str(f) + ".json"
        with open(os.path.join(json_path, json_filename), "wb") as json_file:
            try:
                json.dump(jsondic, json_file)
            except Exception as e:
                warnings.warn(str(e))
                result.append(None)
            finally:
                print "Written " + json_filename
                print ""
                result.append(json_filename)
    return result


if __name__ == "__main__":

    Corpora = {"SemEval14LaptopsTrain": SemEval14LaptopsTrain,
               "SemEval14RestaurantsTrain": SemEval14RestaurantsTrain,
               "SamsungGalaxyS5": SamsungGalaxyS5}

    desc = "Help for pred_generate_json, a script that takes processed corpora and " \
           "embeddings and generates JSON files for training RNN for domain adaptation on " \
           "aspect-based opinion mining using PRED"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("json_path",
                        help="Absolute path to save/load JSON files")

    subparsers = parser.add_subparsers(help='sub-command help',
                                       dest='sub_parser_name')

    parser_create = subparsers.add_parser('create',
                                          help='Create a JSON dataset for training on source dataset')

    parser_create.add_argument("--sentiment", "-sent",
                               action='store_true',
                               help="To generate IOB tags that include sentiment (TARG+/TARG-/TARG0)")

    parser_create.add_argument('--source', "-s",
                               nargs='*',
                               choices=Corpora,
                               help="Names of corpus to use as source. Allowed values are " + ', '.join(Corpora),
                               metavar='')

    parser_create.add_argument('--target', "-t",
                               nargs='*',
                               default=SamsungGalaxyS5,
                               choices=Corpora,
                               help="Names of corpus to use as target. Allowed values are " + ', '.join(Corpora),
                               metavar='')

    embs = ["GoogleNews", "SennaEmbeddings", "WikiDeps"]

    parser_create.add_argument('--embeddings', "-e",
                               nargs='*',
                               choices=embs,
                               help="Names of embeddings to use. Allowed values are " + ', '.join(embs),
                               metavar='')

    parser_add = subparsers.add_parser('add',
                                       help='Obtain the PRED features from trained model and add them')

    parser_add.add_argument('model_path',
                            help="Model path to reload trained model")

    parser_add.add_argument('--folds', "-f",
                            type=int,
                            default=10,
                            help="Number of folds to use. Default=10")

    args = parser.parse_args()

    feat_funcs = [lambda t: "JJ" in t.pos,
                  lambda t: "NN" in t.pos,
                  lambda t: "RB" in t.pos,
                  lambda t: "VB" in t.pos,
                  lambda t: t.iob == "B-NP",
                  lambda t: t.iob == "B-PP",
                  lambda t: t.iob == "B-VP",
                  lambda t: t.iob == "B-ADJP",
                  lambda t: t.iob == "B-ADVP",
                  lambda t: t.iob == "I-NP",
                  lambda t: t.iob == "I-PP",
                  lambda t: t.iob == "I-VP",
                  lambda t: t.iob == "I-ADJP",
                  lambda t: t.iob == "I-ADVP"]

    if not os.path.exists(args.json_path):
        print "Creating " + args.json_path
        os.makedirs(args.json_path)

    if args.sub_parser_name == "create":

        if args.source is None:
            source_names = Corpora.keys()
        else:
            source_names = args.source

        target_names = args.target

        if args.embeddings:
            embeddings_list = []
            if "GoogleNews" in args.embeddings:
                embeddings_list.append(GoogleNews)
            if "SennaEmbeddings" in args.embeddings:
                embeddings_list.append(SennaEmbeddings)
            if "WikiDeps" in args.embeddings:
                embeddings_list.append(WikiDeps)
        else:
            embeddings_list = [GoogleNews, SennaEmbeddings, WikiDeps]

        for Embeddings in embeddings_list:
            print "loading " + str(Embeddings.__name__)
            embeddings = Embeddings()
            for source_name in source_names:
                Source = Corpora[source_name]
                for pipeline in Source.list_frozen():
                    if any(["Chunker" in item for item in pipeline]):
                        source = Source.unfreeze(pipeline)
                        print "Using " + source.name + " " + str(source.pipeline) + ' as source'
                        for target_name in target_names:
                            Target = Corpora[target_name]
                            for pipeline in Target.list_frozen():
                                if any(["Chunker" in item for item in pipeline]):
                                    target = Target.unfreeze(pipeline)
                                    print "Using " + target.name + " " + str(target.pipeline) + ' as target'
                                    build_json_pred_source(source,
                                                           target,
                                                           embeddings,
                                                           args.json_path,
                                                           feat_funcs=feat_funcs,
                                                           sentiment=args.sentiment)

    elif args.sub_parser_name == "add":
        build_json_pred_target(args.json_path,
                               args.model_path,
                               folds=args.folds)
