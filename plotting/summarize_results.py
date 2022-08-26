import matplotlib.pyplot as plt

import argparse
import numpy as np
import sys, os
import pickle
import scipy.stats
import sklearn.metrics
import seaborn as sns
from tests import TESTS

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tfs', type=os.path.abspath, help='List of TFs')
    parser.add_argument('workdir', type=os.path.abspath, help='workdir')
    parser.add_argument('-o',help='Output directory')
    return parser

def load_res(pkl):
    return pickle.load(open(pkl,'rb'))

def main(args):
    args.tfs = open(args.tfs,'r').readlines()
    args.tfs = [x.strip() for x in args.tfs]
    
    pear = []
    auc = []
    pr = []
    for tf in args.tfs:
        res = load_res('{}/{}/results_combined.pkl'.format(args.workdir,tf))
        pear.append(res['pearson'])
        auc.append(res['auc'])
        pr.append(res['pr'])

    pear = np.asarray(pear)
    auc = np.asarray(auc)
    pr = np.asarray(pr)

    score = pear+auc+pr
    best = np.argmax(score, axis=1)
    N = len(args.tfs)
    max_pear = pear[range(N),best]
    max_auc= auc[range(N),best]
    max_pr = pr[range(N),best]

    print(best)
    i = 0
    for b,tf in zip(best, args.tfs):
        print(tf, TESTS[b]['motif_detectors'], TESTS[b]['motif_len'], TESTS[b]['fc_nodes'],max_pear[i],max_auc[i], max_pr[i])
        i += 1
    print('Mean pearson: {:.3f}'.format(np.mean(max_pear)))
    print('Mean AUC: {:.3f}'.format(np.mean(max_auc)))
    print('Mean PR: {:.3f}'.format(np.mean(max_pr)))
    print('Median pearson: {:.3f}'.format(np.median(max_pear)))
    print('Median AUC: {:.3f}'.format(np.median(max_auc)))
    print('Median PR: {:.3f}'.format(np.median(max_pr)))
    

    plt.figure(1)
    sns.heatmap(pear, yticklabels=args.tfs, xticklabels=False)
    plt.yticks(rotation='horizontal')
    plt.xlabel('hyperparameter sweep trial')
    plt.ylabel('TF')
    plt.title('Pearson correlation')
    plt.tight_layout()
    plt.savefig('{}/all_pearson.png'.format(args.o))

    plt.figure(2)
    sns.heatmap(auc, yticklabels=args.tfs, xticklabels=False)
    plt.yticks(rotation='horizontal')
    plt.xlabel('hyperparameter sweep trial')
    plt.ylabel('TF')
    plt.title('AUC')
    plt.tight_layout()
    plt.savefig('{}/all_auc.png'.format(args.o))

    plt.figure(3)
    sns.heatmap(auc, yticklabels=args.tfs, xticklabels=False)
    plt.yticks(rotation='horizontal')
    plt.xlabel('hyperparameter sweep trial')
    plt.ylabel('TF')
    plt.title('Average Precision Recall')
    plt.tight_layout()
    plt.savefig('{}/all_pr.png'.format(args.o))

    plt.figure(4)
    plt.hist(max_pear)
    plt.xlabel('Pearson correlation')
    plt.figure(5)
    plt.hist(max_auc)
    plt.xlabel('AUC')
    plt.figure(6)
    plt.hist(max_pr)
    plt.xlabel('Average Precision Recall')

    results = dict(
        tfs=args.tfs,
        auc=max_auc,
        pr=max_pr,
        pearson=max_pear,
        best_model=best,
        best_model_args=[TESTS[i] for i in best])
    with open('{}/results.pkl'.format(args.o),'wb') as f:
        pickle.dump(results,f)


if __name__ == '__main__':
    main(parse_args().parse_args())
