import matplotlib.pyplot as plt

import argparse
import numpy as np
import sys, os
import pickle
import sklearn.metrics

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input')
    parser.add_argument('--title')
    parser.add_argument('-o', help='Output')
    return parser

def main(args):
    data = pickle.load(open(args.input,'rb'))
    train_preds = []
    train_actual = []
    for i in range(len(data['train.pred'])):
    	train_preds.append(np.argmax(np.rint(data['train.pred'][i])))
    	train_actual.append(np.argmax(np.rint(data['train.actual'][i])))

    val_preds = []
    val_actual = []
    for i in range(len(data['val.pred'])):
    	val_preds.append(np.argmax(np.rint(data['val.pred'][i])))
    	val_actual.append(np.argmax(np.rint(data['val.actual'][i])))
    	

    with open("opt_undersample_auc.txt","a") as file:
        file.write('\n')
        file.write(str(sklearn.metrics.roc_auc_score(np.array(train_preds),np.array(train_actual))))
        file.write('\t')
        file.write(str(sklearn.metrics.roc_auc_score(np.array(val_preds),np.array(val_actual))))
        file.write('\t')
        file.write(str(sklearn.metrics.average_precision_score(np.array(train_preds),np.array(train_actual))))
        file.write('\t')
        file.write(str(sklearn.metrics.average_precision_score(np.array(val_preds),np.array(val_actual))))

    # print("Training AUROC")
    # print(sklearn.metrics.roc_auc_score(np.array(train_actual),np.array(train_preds)))
    # print("Validation AUROC")
    # #print(val_preds)
    # print(sklearn.metrics.roc_auc_score(np.array(val_actual),np.array(val_preds)))

    # print("Training Average Precision")
    # print(sklearn.metrics.average_precision_score(np.array(train_preds),np.array(train_actual)))
    # print("Validation Average Precision")
    # print(sklearn.metrics.average_precision_score(np.array(val_preds),np.array(val_actual)))
    #plt.plot(fpr_train, tpr_train)
    #plt.show()
    # #plt.plot(data['train.actual'],data['train.pred'],'o', label='train')
    # #plt.plot(data['val.actual'],data['val.pred'],'o', label='val')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.legend(loc='best')
    # if not args.o:
    #     plt.show()
    # else:
    #     if args.title:
    #         plt.title(args.title)
    #     plt.savefig(args.o)

if __name__ == '__main__':
    main(parse_args().parse_args())