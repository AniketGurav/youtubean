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


def build_json_weighted_dataset(from_corpus, to_corpus, embeddings, json_path,
                                feat_funcs=[], sentiment=True, folds=10):
    """
    Generate a JSON file for training a domain adaptation model from SRC to TGT by weighting
    examples on the SRC domain. A flag is added on each SRC example to indicate they should be
    weighted when training, using the provided weight. This method is called WEIGHTED.

    Args:
        from_corpus (Corpus):
            Source corpus object
        to_corpus (Corpus):
            Target corpus object
        embeddings (Embeddings):
            Embeddings object
        feat_funcs (list):
            list of functions to extract binary features from tokens, None if no
            binary features should be extracted (default=None)
        folds (int):
            Number of folds to use for training (default=10)
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

    json_path_name = "WEIGHTED" + \
                     "[" + ".".join(from_corpus.pipeline) + "]" + from_corpus.name + "." + \
                     "[" + ".".join(to_corpus.pipeline) + "]" + to_corpus.name + "." + \
                     embeddings.name

    json_path = os.path.join(json_path, json_path_name)
    if not os.path.isdir(json_path):
        os.makedirs(json_path)

    # add all the data in 'from_corpus' to each fold, set weight=True
    all_train_x, all_train_feat_x, all_train_y, all_train_weights = [], [], [], []

    for i in range(len(from_psentences)):
        all_train_x.append([word2idx[word] for word in from_psentences[i]])
        all_train_feat_x.append(from_features[i].tolist())
        all_train_y.append([label2idx[label] for label in from_plabels[i]])
        all_train_weights.append(True)

    # calcule fold sizes
    foldsize = 1.0 * (len(to_psentences)) / folds
    rest = 1.0 * (len(to_psentences)) % folds

    result = []

    for f in range(1, folds + 1):

        start = (f - 1) * foldsize
        end = f * foldsize

        train_size = foldsize * (folds - 1)

        if f == folds:
            end += rest

        # copy list
        train_x = list(all_train_x)
        train_feat_x = list(all_train_feat_x)
        train_y = list(all_train_y)
        train_weights = list(all_train_weights)

        train_ids, valid_ids, test_ids = [], [], []
        valid_x, valid_feat_x, valid_y, valid_weights = [], [], [], []
        test_x, test_feat_x, test_y, test_weights = [], [], [], []

        for i in range(len(to_psentences)):
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
            train_x.append([word2idx[word] for word in to_psentences[i]])
            train_feat_x.append(to_features[i].tolist())
            train_y.append([label2idx[label] for label in to_plabels[i]])
            train_weights.append(False)

        for i in valid_ids:
            valid_x.append([word2idx[word] for word in to_psentences[i]])
            valid_feat_x.append(to_features[i].tolist())
            valid_y.append([label2idx[label] for label in to_plabels[i]])
            valid_weights.append(False)

        for i in test_ids:
            test_x.append([word2idx[word] for word in to_psentences[i]])
            test_feat_x.append(to_features[i].tolist())
            test_y.append([label2idx[label] for label in to_plabels[i]])
            test_weights.append(False)

        jsondic = dict()
        jsondic["featdim"] = len(feat_funcs)
        jsondic["fold"] = f
        jsondic["train_x"], jsondic["train_feat_x"], jsondic["train_y"] = train_x, train_feat_x, train_y
        jsondic["valid_x"], jsondic["valid_feat_x"], jsondic["valid_y"] = valid_x, valid_feat_x, valid_y
        jsondic["test_x"], jsondic["test_feat_x"], jsondic["test_y"] = test_x, test_feat_x, test_y

        jsondic["domain"] = True
        jsondic["train_weights"], jsondic["valid_weights"], jsondic["test_weights"] = train_weights, \
                                                                                      valid_weights, \
                                                                                      test_weights
        jsondic["word2idx"] = word2idx
        jsondic["label2idx"] = label2idx
        jsondic["embeddings"] = matrix.tolist()

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

    desc = "Help for weighted_generate_json, a script that takes processed corpora and " \
           "embeddings and generates JSON files for training RNN for domain adaptation aspect-based " \
           "opinion mining using weighted examples"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("json_path",
                        help="Absolute path to store JSON files")

    parser.add_argument("--sentiment", "-sent",
                        action='store_true',
                        help="To generate IOB tags that include sentiment (TARG+/TARG-/TARG0)")

    parser.add_argument('--source', "-s",
                        nargs='*',
                        choices=Corpora,
                        help="Names of corpus to use as source. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    parser.add_argument('--target', "-t",
                        nargs='*',
                        default=SamsungGalaxyS5,
                        choices=Corpora,
                        help="Names of corpus to use as target. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    embs = ["GoogleNews", "SennaEmbeddings", "WikiDeps"]

    parser.add_argument('--embeddings', "-e",
                        nargs='*',
                        choices=embs,
                        help="Names of embeddings to use. Allowed values are " + ', '.join(embs),
                        metavar='')

    parser.add_argument('--folds', "-f",
                        type=int,
                        default=10,
                        help="Number of folds to use. Default=10")

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

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print "Creating " + args.json_path
        os.makedirs(args.json_path)

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
                                build_json_weighted_dataset(source,
                                                            target,
                                                            embeddings,
                                                            args.json_path,
                                                            feat_funcs=feat_funcs,
                                                            folds=args.folds,
                                                            sentiment=args.sentiment)
