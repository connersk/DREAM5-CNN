'''Skeleton script'''

import argparse
import numpy as np
import sys, os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tests import TESTS

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pkl', help='Input')
    parser.add_argument('--tf', help='TF name')
    parser.add_argument('-o', help='Output')
    return parser

def get_test_index(md, ml):
    for i,t in TESTS.items():
        if t['motif_detectors'] == md:
            if t['motif_len'] == ml:
                if t['weight_decay'] == 0.001:
                    if t['dropout'] == 0.5:
                        return i
    raise RuntimeError

def main(args):
    res = pickle.load(open(args.pkl,'rb'))
    auc = np.asarray(res['auc'])
    pr = np.asarray(res['pr'])
    pear = np.asarray(res['pearson'])
    scores = auc + pr + pear
    
    MOTIF_DETECTORS= [16,32,64,128]
    MOTIF_LEN = [4,8,12,24]

    score = np.zeros((4,4))
    for i,md in enumerate(MOTIF_DETECTORS):
        for j,ml in enumerate(MOTIF_LEN):
            k = get_test_index(md,ml)
            score[i,j] = scores[k]
    sns.heatmap(score, xticklabels=MOTIF_LEN, yticklabels=MOTIF_DETECTORS, vmin=0, vmax=3)
    plt.xticks(rotation='horizontal')
    plt.xlabel('motif length')
    plt.ylabel('motif detectors')
    plt.title('{}\n CNN performance (AUC+PR+pearson r)'.format(args.tf))
    if args.o:
        plt.savefig(args.o)
    else:
        plt.show()
    

if __name__ == '__main__':
    main(parse_args().parse_args())
