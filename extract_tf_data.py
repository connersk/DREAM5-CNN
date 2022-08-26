import sys
import os
import argparse
import numpy as np
import sys, os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', type=os.path.abspath, default='DREAM5_PDB_Data_TrainingSet.txt',
        help='Input DREAM5 training set file')
    parser.add_argument('-tf', default='Cebpb', help='%(default)s')
    parser.add_argument('--array-type', choices=('HK','ME'), nargs='+', default='all')
    parser.add_argument('-o', help='Output file')
    return parser

def quality_control(data):
    '''Discard lines if flag != 0'''
    lines = [x for x in data if x[-1].strip() == '0']
    return lines

def extract_data(infile, tf, array_types, outfile):
    with open(infile,'r') as f:
        lines = []
        for line in f.readlines():
            if line.split('\t')[0] == tf:
                lines.append(line)
    lines = [x.split('\t') for x in lines]
    if array_types != 'all':
        lines = [x for x in lines if x[1] in array_types]
    print('{} entries found for TF {}'.format(len(lines),tf))
    lines = quality_control(lines)
    print('{} entries remaining after QC'.format(len(lines)))
    if os.path.exists(outfile):
        print('WARNING: file {} already exists. overwriting'.format(outfile))
    with open(outfile,'w') as f:
        for line in lines:
            #intensity = float(line[5])-float(line[6]) # median signal intensity - median background intensity
            f.write('\t'.join([line[2][0:35],line[5], line[6]])) # seq, median signal, median background
            f.write('\n')
    print('Wrote {}'.format(outfile))

    #x = np.array([float(line[5]) - float(line[6]) for line in lines])
    #plt.plot(range(len(x)), x,'o')
    #plt.show()


def main(args):
    extract_data(args.i, args.tf, args.array_type, args.o)

if __name__ == '__main__':
    main(parse_args().parse_args())
