import matplotlib.pyplot as plt

import argparse
import numpy as np
import sys, os
import pickle
import scipy.stats
import sklearn.metrics
from tests import TESTS

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tf_dir', type=os.path.abspath, help='Folder containing test subdirectories')
    parser.add_argument('-o')
    return parser

def mse(x1,x2):
    return np.mean((x1-x2)**2)

def pearsonr(x1,x2):
    return scipy.stats.pearsonr(x1,x2)[0][0]

def spearmanr(x1,x2):
    return scipy.stats.spearmanr(x1,x2)[0]

def compute_auc(actual, pred):
    return sklearn.metrics.roc_auc_score(actual, pred)

def compute_pr(actual, pred):
    return sklearn.metrics.average_precision_score(actual,pred)

def summarize(actual, pred, m=None, s=None):
    err = mse(actual,pred)
    pear = pearsonr(actual,pred)
    spear = spearmanr(actual,pred)
    if m is None:
        return err, pear, spear, None, None
    actual *= s
    actual += m
    actual = np.exp(actual)
    actual_class = np.zeros(actual.shape)
    w = np.where(actual > np.mean(actual)+3*np.std(actual))
    print('{} true positives'.format(len(w[0])))
    if len(w[0]) == 0:
        return err, pear, spear, None, None
    actual_class[w] = 1.0
    auc = compute_auc(actual_class,pred)
    pr = compute_pr(actual_class,pred)
    return err, pear, spear, auc, pr

def main(args):
    all_mse = []
    all_pear = []
    all_spear = []
    all_auc = []
    all_pr = []
    for test in range(len(TESTS)):
        infile = '{}/{}/{}.pkl'.format(args.tf_dir, test, os.path.basename(args.tf_dir))
        data = pickle.load(open(infile,'rb'))
        m,s = data.get('val.unnormalized',(None,None))
        err, pear, spear, auc, pr = summarize(data['val.actual'],data['val.pred'], m,s)
        all_mse.append(err)
        all_pear.append(pear)
        all_spear.append(spear)
        all_auc.append(auc)
        all_pr.append(pr)
    all_auc = [x for x in all_auc if x]
    all_pr= [x for x in all_pr if x]
    print('Combined results')
    print('Average MSE: {:.3f}'.format(np.mean(all_mse)))
    print('Average pearson r: {:.3f}'.format(np.mean(all_pear)))
    print('Average spearman r: {:.3f}'.format(np.mean(all_spear)))
    print('Average AUC: {:.3f}'.format(np.mean(all_auc)))
    print('Average PR: {:.3f}'.format(np.mean(all_pr)))

    print('Median MSE: {:.3f}'.format(np.median(all_mse)))
    print('Median pearson r: {:.3f}'.format(np.median(all_pear)))
    print('Median spearman r: {:.3f}'.format(np.median(all_spear)))
    print('Median AUC: {:.3f}'.format(np.median(all_auc)))
    print('Median PR: {:.3f}'.format(np.median(all_pr)))

    def print_i(i):
        print(i)
        print(TESTS[i])
        print(all_mse[i])
        print(all_pear[i])
        print(all_spear[i])
        print(all_auc[i])
        print(all_pr[i])
    print('Best MSE')
    print_i(np.argmin(all_mse))
    print('Best Pearson')
    print_i(np.argmax(all_pear))
    print('Best Spearman')
    print_i(np.argmax(all_spear))
    print('Best AUC')
    print_i(np.argmax(all_auc))
    print('Best PR')
    print_i(np.argmax(all_pr))

    plt.plot(range(len(TESTS)), all_pear, 'o', label='pearson')
    #plt.plot(range(len(TESTS)), all_spear, 'o', label='spearman')
    plt.plot(range(len(TESTS)), all_auc, 'o', label='AUC')
    plt.plot(range(len(TESTS)), all_pr, 'o', label='PR')
    #plt.plot(range(len(TESTS)), all_mse, 'o', label='MSE')
    plt.xlabel('test')
    plt.ylabel('performance')
    plt.legend(loc='best')

    if not args.o:
        plt.show()
    else:
        plt.savefig(args.o)
        results = dict(pearson=all_pear,
                        spearman=all_spear,
                        mse=all_mse,
                        auc=all_auc,
                        pr=all_pr)
        pickle.dump(results,open('{}.pkl'.format(os.path.splitext(args.o)[0]),'wb'))

if __name__ == '__main__':
    main(parse_args().parse_args())
