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


    x_vals = np.array([float(line[5]) - float(line[6]) for line in lines])
    four_sd = 4*np.std(x_vals)
    high_vals = np.extract(x_vals> four_sd, x_vals)

    if os.path.exists(outfile):
        print('WARNING: file {} already exists. overwriting'.format(outfile))
    with open(outfile,'w') as f:
        for line in lines:
            intensity = float(line[5])-float(line[6]) # median signal intensity - median background intensity
            f.write('\t'.join([line[2][0:35],str(intensity)]))


            if float(line[5])-float(line[6]) in high_vals:
                f.write('\t')
                f.write(str(1))
            else:
                f.write('\t')
                f.write(str(0))


            f.write('\n')
    print('Wrote {}'.format(outfile))

    x = np.array([float(line[5]) - float(line[6]) for line in lines])
    four_sd = 4*np.std(x)
    high_vals = np.extract(x> four_sd, x)
    #out
    #plt.plot(range(len(x)), x,'o')
    #plt.plot(range(len(high_vals)), high_vals,'o',c="r")
    #plt.show()


def main(args):
    extract_data(args.i, args.tf, args.array_type, args.o)

if __name__ == '__main__':
    main(parse_args().parse_args())