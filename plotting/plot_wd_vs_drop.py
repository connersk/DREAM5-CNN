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

def get_test_index(dr, wd):

    for i in range(48,64):
        t = TESTS[i]
        if t['dropout'] == dr:
            if t['weight_decay'] == wd:
                return i
    raise RuntimeError

def main(args):
    res = pickle.load(open(args.pkl,'rb'))
    auc = np.asarray(res['auc'])
    pr = np.asarray(res['pr'])
    pear = np.asarray(res['pearson'])
    scores = auc + pr + pear
    
    DROPOUT = [0,.25,.5]
    W_DECAY = [0,.0001,.001,.01,.1]

    score = np.zeros((3,5))
    for i,dr in enumerate(DROPOUT):
        for j,wd in enumerate(W_DECAY):
            k = get_test_index(dr, wd)
            score[i,j] = scores[k]
    sns.heatmap(score, xticklabels=W_DECAY, yticklabels=DROPOUT, vmin=0, vmax=3)
    plt.xticks(rotation='horizontal')
    plt.xlabel('Weight Decay')
    plt.ylabel('Dropout')
    plt.title('{}\n CNN performance (AUC+PR+pearson r)'.format(args.tf))
    if args.o:
        plt.savefig(args.o)
    else:
        plt.show()
    

if __name__ == '__main__':
    main(parse_args().parse_args())
