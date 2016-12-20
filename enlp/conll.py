#!/usr/bin/python
# -*- coding: utf-8 -*-

from .settings import CONLLEVAL_PATH

import numpy
import random
import os
import subprocess


def conlleval(p, g, w, filename):
    """
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    """
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)


def get_perf(filename):
    """
    run conlleval.pl perl script to obtain
    precision/recall and F1 score
    """

    proc = subprocess.Popen(["perl", CONLLEVAL_PATH],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision,
            'r': recall,
            'f1': f1score}
