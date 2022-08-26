import numpy as np
import pickle
import argparse
import os, sys
import sklearn.metrics
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import FloatTensor

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

    group = parser.add_argument_group('Training parameters')
    group.add_argument('--min-epochs', type=int, default=5, help='Min epochs')
    group.add_argument('--max-epochs', type=int, default=25, help='Max epochs')
    group.add_argument('--batchsize', type=int, default=64,
            help='Batch size')
    group.add_argument('--learning-rate', type=float, default=0.001,
            help='Learning rate for Adam optimizer')
    group.add_argument('--weight-decay', type=float, default=0.001,
            help='Weight decay for Adam optimizer')
    group.add_argument('--incl-rev-comp', action='store_true',
            help='Include the reverse complement strand in training data')

    group = parser.add_argument_group('Architecture parameters')
    group.add_argument('--motif-detectors', type=int, default=64, help='Number of motif detectors')
    group.add_argument('--motif-len', type=int, default=24, help='Motif length')
    group.add_argument('--fc-nodes', type=int, default=32, help='Number of nodes in fully connected layer')
    group.add_argument('--dropout', type=float, default=0.5, help='Probability of zeroing out element in dropout layer')
    group.add_argument('--l1-reg',action='store_true',help='Use L1 regularization, strength is --weight-decay parameter')

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


def load_data(infile, pad=0, include_reverse_complement=False):
    '''Returns sequence encoded in a 4xN matrix, probe intensity'''
    with open(infile) as f:
        lines = f.readlines()
    lines = [x.strip().split('\t') for x in lines] # first col is sequence, second col is intensity
    xs = np.array([encode_seq(x[0], pad) for x in lines])
    ys = np.array([float(x[1]) for x in lines])
    ys = np.log(ys)
    m, s = np.mean(ys), np.std(ys)
    ys -= np.mean(ys)
    ys /= np.std(ys)
    ys = ys.reshape((len(ys),1))

    if include_reverse_complement:
        x_revc = np.array([encode_seq(revcomp(x[0]), pad) for x in lines])
        xs = np.vstack((xs, x_revc))
        ys = np.vstack((ys,ys))
    return xs, ys, m, s

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
    def __init__(self, motif_detectors=64, motif_len=24, fc_nodes=32, dropout=0.5):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(4, motif_detectors, motif_len)
        self.fc = nn.Linear(motif_detectors,fc_nodes)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(fc_nodes,1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.max(dim=2)[0]  # max returns (max_values, max_indices)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

def train_epoch(model, xdata, ydata, opt, MSELoss, batchsize):
    xdata, ydata = sklearn.utils.shuffle(xdata,ydata)
    xbatched = split_batches(xdata,batchsize)
    ybatched = split_batches(ydata,batchsize)
    for x, y in zip(xbatched,ybatched):
        model.zero_grad()
        sequences = Variable(torch.from_numpy(x).float())
        y = Variable(torch.from_numpy(y).float())
        pred = model(sequences)
        loss = MSELoss(pred, y)
        loss.backward()
        opt.step()

def main(args):
    xdata, ydata, mm_t, ss_t = load_data(args.train, args.motif_len-1, args.incl_rev_comp)
    xval, yval, mm_v, ss_v = load_data(args.val, args.motif_len-1)
    xval_, yval_ = Variable(FloatTensor(xval)), Variable(FloatTensor(yval))

    model = Model(args.motif_detectors, args.motif_len, args.fc_nodes, args.dropout)
    if args.l1_reg:
        mse_only = nn.MSELoss()
        def MSELoss(pred, actual):
            loss = mse_only(pred,actual)
            for p in model.parameters():
                loss += args.weight_decay*p.abs().mean()
            return loss
        opt = optim.Adam(lr=args.learning_rate,weight_decay=0.0, params=model.parameters()) # set L2 reg to 0
    else:
        MSELoss = nn.MSELoss()
        opt = optim.Adam(lr=args.learning_rate,weight_decay=args.weight_decay, params=model.parameters())
    for epoch in range(args.min_epochs):
        train_epoch(model, xdata, ydata, opt, MSELoss, args.batchsize)
        model.eval()
        loss = MSELoss(model(xval_),yval_)
        print('Epoch {}, Validation MSE={}'.format(epoch, loss.data[0]))
        model.train()
    weights = model.state_dict()

    while 1:
        epoch += 1
        if epoch > args.max_epochs:
            break
        train_epoch(model, xdata, ydata, opt, MSELoss, args.batchsize)
        model.eval()
        next_loss = MSELoss(model(xval_),yval_)
        model.train()
        print('Epoch {}, Validation MSE={}'.format(epoch, next_loss.data[0]))
        if next_loss.data[0] > loss.data[0]:
            break
        loss = next_loss
        weights = copy.deepcopy(model.state_dict()) 
    model.load_state_dict(weights) # weights from previous iteration with lowest validation loss

    model.eval()
    print('Training Performance')
    pred = model(Variable(FloatTensor(xdata)))
    loss = MSELoss(pred, Variable(FloatTensor(ydata)))
    print('MSE: {}'.format(loss.data[0]))
    print('r2: {}'.format(sklearn.metrics.r2_score(ydata,pred.data.numpy())))

    results = {'train.actual': ydata,
               'train.pred': pred.data.numpy()}

    print('Validation Performance')
    pred = model(xval_)
    loss = MSELoss(pred, yval_)
    print('MSE: {}'.format(loss.data[0]))
    print('r2: {}'.format(sklearn.metrics.r2_score(yval,pred.data.numpy())))

    results['val.actual'] = yval
    results['val.pred'] = pred.data.numpy()
    results['provenance'] = sys.argv
    results['epochs'] = epoch

    # unnormalized mean and std
    results['train.unnormalized'] = (mm_t,ss_t)
    results['val.unnormalized'] = (mm_v,ss_v)

    with open(args.o,'wb') as f:
        pickle.dump(results,f)
        pickle.dump(model.state_dict(),f)

if __name__ == '__main__':
    main(parse_args().parse_args())
