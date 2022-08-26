import numpy as np
import pickle
import argparse
import os, sys
import sklearn.metrics
import sklearn.utils


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import FloatTensor
import torchvision
import torchvision.transforms as transforms

np.random.seed(0)

ALPHABET = ['A','C','G','T']
A_TO_INDEX = {xx:ii for ii,xx in enumerate(ALPHABET)} # dict mapping 'A' -> 0, 'C' -> 1, etc.
WCPAIR = {'A':'T',
          'T':'A',
          'G':'C',
          'C':'G'}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train', type=os.path.abspath, help='Input file')
    parser.add_argument('val', type=os.path.abspath, help='Input file')
    parser.add_argument('-o', default='tmp_results.pkl', help='Output pickle with results')
    parser.add_argument('--one_array', action='store_true',help='performing trainig and testing on only 1st array type')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('--nepochs', type=int, default=5, help='Num epochs')
    #group.add_argument('--min-epochs', type=int, default=5, help='Min epochs')
   # group.add_argument('--max-epochs', type=int, default=25, help='Max epochs')
    group.add_argument('--batchsize', type=int, default=64,
            help='Batch size')
    group.add_argument('--incl-rev-comp',action='store_true',
            help='Include the reverse complement strand in training data')

    group = parser.add_argument_group('Architecture parameters')
    group.add_argument('--motif-detectors', type=int, default=64, help='Number of motif detectors')
    group.add_argument('--motif-len', type=int, default=24, help='Motif length')
    group.add_argument('--fc-nodes', type=int, default=32, help='Number of nodes in fully connected layer')
    #group.add_argument('--dropout', type=int, default=0.5, help='Probability of zeroing out element in dropout layer')

    group = parser.add_argument_group('Sampling correction')
    group.add_argument('--sampling', type=str,choices=('undersampling','oversampling',''), default='', help='Sampling correction for imbalanced classes')
    group.add_argument('--sampling_factor', type=int, default = 5, help='Factor to over or under sample by')

    return parser

def encode_seq(seq, pad=0):
    assert set(seq) <= set(ALPHABET), set(seq)
    m = np.zeros((4,len(seq)))
    for ii, a in enumerate(seq):
        m[A_TO_INDEX[a], ii] = 1
    if pad:
        m = np.hstack((.25*np.ones((4,pad)),m,.25*np.ones((4,pad))))
    return m

def revcomp(seq):
    '''return the reverse complement of an input sequence'''
    assert set(seq) <= set(ALPHABET), set(seq)
    return ''.join([WCPAIR[a] for a in seq[::-1]])


def load_data(infile, pad=0, include_reverse_complement=False, sampling="", sampling_factor=5, train=False):
    '''Returns sequence encoded in a 4xN matrix, probe intensity'''
    with open(infile) as f:
        lines = f.readlines()
    lines = [x.strip().split('\t') for x in lines] # first col is sequence, second col is intensity
    xs = np.array([encode_seq(x[0], pad) for x in lines])
    # just looking at the classification problem for now

    def one_hot_encoder(x):
        if x == 0:
            return np.array([1.0,0.0])
        else:
            return np.array([0.0,1.0])

    #ys = np.array([one_hot_encoder(float(x[2])) for x in lines])
    ys = np.array([float(x[1]) for x in lines])
    four_sd = 4*np.std(ys)
    new_ys = np.zeros(len(ys))
    for i in range(len(ys)):
        if ys[i] >= four_sd:
            new_ys[i] = 1
    ys = new_ys
    ys = np.array([[one_hot_encoder(float(x)) for x in ys]])
    ys = ys[0]

    # print(ys)
    # print(len(ys))
    # print(len(xs))

    if include_reverse_complement:
        x_revc = np.array([encode_seq(revcomp(x[0]), pad) for x in lines])
        xs = np.vstack((xs, x_revc))
        ys = np.vstack((ys,ys))

    # calculate number of true and false samples, find indices for both
    num_true = 0
    num_false = 0
    false_index = []
    true_index = []
    for i in range(len(ys)):
        if np.array_equal(ys[i,:],np.array([0.0,1.0])):
            num_true += 1
            true_index.append(i)
        else:
            num_false +=1
            false_index.append(i)

    true_xs = xs[true_index,:]
    true_ys = ys[true_index,:]

    false_xs = xs[false_index,:]
    false_ys = ys[false_index,:]


    print("=========Before Sampling========")
    print('Num high binders: {}'.format(len(true_xs)))
    print('Num low binders: {}'.format(len(false_xs)))


    if sampling == "undersampling":
        num_low_binders =  num_true * sampling_factor
        low_binders_index = np.random.choice(false_index, num_low_binders)
        false_xs = xs[low_binders_index,:]
        false_ys = ys[low_binders_index,:]

    if sampling == "oversampling":
        num_repeats = sampling_factor
        true_xs = np.repeat(xs[true_index,:],[num_repeats]*len(xs[true_index,:]),axis=0)
        true_ys = np.repeat(ys[true_index,:],[num_repeats]*len(ys[true_index,:]),axis=0)

    print("=========After Sampling========")
    print('Num high binders: {}'.format(len(true_xs)))
    print('Num low binders: {}'.format(len(false_xs)))

    xs = np.vstack((true_xs,false_xs))
    ys = np.vstack((true_ys,false_ys))

    xs, ys = sklearn.utils.shuffle(xs, ys)

    if train:
        return xs, ys, float(num_false), float(num_true)
    else: 
        return xs, ys

def load_data_one_array(infile, pad=0, include_reverse_complement=False, sampling="", sampling_factor=5, train=False):
    '''Returns sequence encoded in a 4xN matrix, probe intensity'''
    with open(infile) as f:
        lines = f.readlines()
    lines = [x.strip().split('\t') for x in lines] # first col is sequence, second col is intensity
    xs = np.array([encode_seq(x[0], pad) for x in lines])
    # just looking at the classification problem for now

    def one_hot_encoder(x):
        if x == 0:
            return np.array([1.0,0.0])
        else:
            return np.array([0.0,1.0])

    # ys = np.array([one_hot_encoder(float(x[2])) for x in lines])
    ys = np.array([float(x[1]) for x in lines])
    four_sd = 4*np.std(ys)
    new_ys = np.zeros(len(ys))
    for i in range(len(ys)):
        if ys[i] >= four_sd:
            new_ys[i] = 1
    ys = new_ys
    ys = np.array([[one_hot_encoder(float(x)) for x in ys]])
    ys = ys[0]

    if include_reverse_complement:
        x_revc = np.array([encode_seq(revcomp(x[0]), pad) for x in lines])
        xs = np.vstack((xs, x_revc))
        ys = np.vstack((ys,ys))

    # calculate number of true and false samples, find indices for both
    num_true = 0
    num_false = 0
    false_index = []
    true_index = []
    for i in range(len(ys)):
        if np.array_equal(ys[i,:],np.array([0.0,1.0])):
            num_true += 1
            true_index.append(i)
        else:
            num_false +=1
            false_index.append(i)

    true_xs = xs[true_index,:]
    true_ys = ys[true_index,:]

    false_xs = xs[false_index,:]
    false_ys = ys[false_index,:]


    print("=========Before Sampling========")
    print('Num high binders: {}'.format(len(true_xs)))
    print('Num low binders: {}'.format(len(false_xs)))


    if sampling == "undersampling":
        num_low_binders =  num_true * sampling_factor
        low_binders_index = np.random.choice(false_index, num_low_binders)
        false_xs = xs[low_binders_index,:]
        false_ys = ys[low_binders_index,:]

    if sampling == "oversampling":
        num_repeats = sampling_factor
        true_xs = np.repeat(xs[true_index,:],[num_repeats]*len(xs[true_index,:]),axis=0)
        true_ys = np.repeat(ys[true_index,:],[num_repeats]*len(ys[true_index,:]),axis=0)

    print("=========After Sampling========")
    print('Num high binders: {}'.format(len(true_xs)))
    print('Num low binders: {}'.format(len(false_xs)))

    num_true_train = int(np.ceil(0.75*len(true_xs)))
    train_true_index = np.random.choice(len(true_xs),num_true_train,replace=False)
    val_true_index = np.array([i for i in range(len(true_xs)) if i not in train_true_index])

    num_false_train = int(np.ceil(0.75*len(false_xs)))
    train_false_index = np.random.choice(len(false_xs),num_false_train,replace=False)
    val_false_index = np.array([i for i in range(len(false_xs)) if i not in train_false_index])

    train_xs = np.vstack((true_xs[train_true_index,:],false_xs[train_false_index,:]))
    train_ys = np.vstack((true_ys[train_true_index,:],false_ys[train_false_index,:]))

    val_xs = np.vstack((true_xs[val_true_index,:],false_xs[val_false_index,:]))
    val_ys = np.vstack((true_ys[val_true_index,:],false_ys[val_false_index,:]))

    train_xs, train_ys = sklearn.utils.shuffle(train_xs, train_ys)
    val_xs, val_ys = sklearn.utils.shuffle(val_xs, val_ys)

    print("=========After Train/Test Split========")
    print('Num train: {}'.format(len(train_xs)))
    print('Num test: {}'.format(len(val_xs)))

    # return train_xs, train_ys, val_xs, val_ys

    if train:
        return train_xs, train_ys, val_xs, val_ys, float(num_false), float(num_true)
    else: 
        return train_xs, train_ys, val_xs, val_ys

def split_batches(data, batchsize):
    '''
    Break up data into batches

    Inputs:
        data (list): input data (list of datapoints given by load_data())
        batchsize (int): size of each batch

    Output:
        Chunked data (list)
    '''
    nbatches = int(np.ceil(len(data)/float(batchsize)))
    batches = []
    for i in range(nbatches):
        batches.append(data[i*batchsize:(i+1)*batchsize])
    return batches

class Model(nn.Module):
    def __init__(self,motif_detectors=64, motif_len=24, fc_nodes=32):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(4, motif_detectors, motif_len)
        self.fc = nn.Linear(motif_detectors,fc_nodes)
        #self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(fc_nodes,2)

       # self.conv = nn.Conv1d(4, 16, 8) # dropping down kernel size from 24 to 8
        #self.fc = nn.Linear(16,32)
        #self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.max(dim=2)[0]  # max returns (max_values, max_indices)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.output(x)
        # could include dropout here if I want to 
        x = F.softmax(x)
        return x

def main(args):


    epochs = args.nepochs
    xdata, ydata, num_low, num_high = load_data(args.train, args.motif_len-1, args.incl_rev_comp, args.sampling, args.sampling_factor, True)
    xbatched = split_batches(xdata,args.batchsize)
    ybatched = split_batches(ydata,args.batchsize)
    # if this doesn't work, will see how adding rev comp affects
    xval, yval = load_data(args.val, args.motif_len-1)

    model = Model()
    # Going to try scaling all data by weights for the different variabeles
    # bce_only = nn.BCELoss()
    # def BCELoss(pred, actual, weights):
    #     loss = bce_only(pred,actual)
    #     for p in model.parameters():
    #         loss += args.weight_decay*p.abs().mean()
    #     return loss
    #weights  = FloatTensor(np.array([1/num_low, 1/num_high]))
    #weights  = FloatTensor(np.array([num_low, num_high]))#trying inverted weights, should be 1/
            #hmm ok, so this sorta seems to be working out nicely
            # can also try swapping them, that would make more sense to me
    #weights  = FloatTensor(np.array([(num_low + num_high)/(num_low), (num_low + num_high)/(num_high)])) # need to reverse this
    #print("=====Class Weights=====")
    #print(weights)
   # BCELoss = nn.BCELoss(weight = weights)
    BCELoss = nn.BCELoss()
    opt = optim.Adam(lr=0.001,weight_decay=0.001, params=model.parameters())
    for epoch in range(epochs):
        xdata, ydata = sklearn.utils.shuffle(xdata, ydata)
        xbatched = split_batches(xdata, args.batchsize)
        ybatched = split_batches(ydata, args.batchsize)
        for x, y in zip(xbatched,ybatched):
            model.zero_grad()
            sequences = Variable(FloatTensor(x))
            y = Variable(FloatTensor(y))
            pred = model(sequences)
            loss = BCELoss(pred, y)
            loss.backward()
            opt.step()
        # switch to model.eval()? how to unswitch
        xval_ = Variable(FloatTensor(xval))
        yval_ = Variable(FloatTensor(yval))
        loss = BCELoss(model(xval_),yval_)
        print('Epoch {}, Validation BCE={}'.format(epoch, loss.data[0]))


    model.eval()
    print('Training Performance')
    pred = model(Variable(FloatTensor(xdata)))
    #print(pred)
    loss = BCELoss(pred, Variable(FloatTensor(ydata)))
    print('BCE: {}'.format(loss.data[0]))
    

    results = {'train.actual': ydata,
               'train.pred': pred.data.numpy()}

    train_preds = []
    train_actual = []
    for i in range(len(results['train.pred'])):
        train_preds.append(np.argmax(np.rint(results['train.pred'][i])))
        train_actual.append(np.argmax(np.rint(results['train.actual'][i])))
    print('Predicted binders: {}'.format(np.sum(np.array(train_preds))))
    print('Actual binders: {}'.format(np.sum(np.array(train_actual))))
    train_AUC = sklearn.metrics.roc_auc_score(np.array(train_preds),np.array(train_actual))
    train_AP = sklearn.metrics.average_precision_score(np.array(train_preds),np.array(train_actual))
    print('AUC: {}'.format(train_AUC))
    print('AP: {}'.format(train_AP))

    print('Validation Performance')
    pred = model(Variable(FloatTensor(xval)))
    loss = BCELoss(pred, Variable(FloatTensor(yval)))
    print('BCE: {}'.format(loss.data[0]))

    results['val.actual'] = yval
    results['val.pred'] = pred.data.numpy()

    val_preds = []
    val_actual = []
    for i in range(len(results['val.pred'])):
        val_preds.append(np.argmax(np.rint(results['val.pred'][i])))
        val_actual.append(np.argmax(np.rint(results['val.actual'][i])))
    print('Predicted binders: {}'.format(np.sum(np.array(val_preds))))
    print('Actual binders: {}'.format(np.sum(np.array(val_actual))))
    val_AUC = sklearn.metrics.roc_auc_score(np.array(val_preds),np.array(val_actual))
    val_AP = sklearn.metrics.average_precision_score(np.array(val_preds), np.array(val_actual))
    print('AUC: {}'.format(val_AUC))
    print('AP: {}'.format(val_AP))


    results['provenance'] = sys.argv
    results['epochs'] = epoch

    with open(args.o,'wb') as f:
        pickle.dump(results,f)
        pickle.dump(model.state_dict(),f)


    ## Need precision recall to do precision recall
if __name__ == '__main__':
    main(parse_args().parse_args())