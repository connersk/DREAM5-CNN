import matplotlib.pyplot as plt

import argparse
import numpy as np
import sys, os
import pickle
import scipy.stats
import sklearn.metrics

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input')
    parser.add_argument('--title')
    parser.add_argument('-o', help='Output')
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
    actual_class[w] = 1.0
    auc = compute_auc(actual_class,pred)
    pr = compute_pr(actual_class,pred)
    return err, pear, spear, auc, pr

def main(args):
    data = pickle.load(open(args.input,'rb'))
    print('Training set')
    m,s = data.get('train.unnormalized',(None,None))
    err, pear, spear, auc, pr = summarize(data['train.actual'],data['train.pred'], m,s)
    print('MSE: {:.3f}'.format(err))
    print('pearson: {:.3f}'.format(pear))
    print('spearman: {:.3f}'.format(spear))
    print('AUC: {:.3f}'.format(auc))
    print('PR: {:.3f}'.format(pr))
    print('Validation set')
    m,s = data.get('val.unnormalized',(None,None))
    err, pear, spear, auc, pr = summarize(data['val.actual'],data['val.pred'], m,s)
    print('MSE: {:.3f}'.format(err))
    print('pearson: {:.3f}'.format(pear))
    print('spearman: {:.3f}'.format(spear))
    print('AUC: {:.3f}'.format(auc))
    print('PR: {:.3f}'.format(pr))
    plt.plot(data['train.actual'],data['train.pred'],'o', label='train')
    plt.plot(data['val.actual'],data['val.pred'],'o', label='val')
    plt.xlabel('actual')
    plt.ylabel('pred')
    plt.legend(loc='best')
    if args.title:
       plt.title(args.title)
    else:
        title = 'Val MSE: {:.3f}, pearson: {:.3f}, spearman: {:.3f}'.format(err, pear, spear)
        if auc:
            title += ' AUC: {:.3f}, PR: {:.3f}'.format(auc, pr)
        plt.title(title)
    if not args.o:
        plt.show()
    else:
        plt.savefig(args.o)

if __name__ == '__main__':
    main(parse_args().parse_args())
