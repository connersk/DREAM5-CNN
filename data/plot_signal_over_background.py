import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns

def load_data(infile):
    with open(infile,'r') as f:
        lines = f.readlines()
    data1 = np.array([float(x.strip().split()[1]) for x in lines])
    data2 = np.array([float(x.strip().split()[2]) for x in lines])
    return data1/data2

def main():
    HK_files = glob.glob('*HK.txt')
    hk_data = [load_data(f) for f in HK_files]
    for d, f in zip(hk_data,HK_files):
        plt.hist(d,bins=20,label=f, alpha=0.5)
        print(min(d))
    plt.title('HK array')
    plt.xlabel('signal median / signal background')
    plt.legend(loc='best')
    plt.show()

    plt.clf()
    ME_files = glob.glob('*ME.txt')
    me_data = [load_data(f) for f in ME_files]
    for d, f in zip(me_data,HK_files):
        plt.hist(d,bins=20,label=f, alpha=0.5)
        print(min(d))
    plt.title('ME array')
    plt.xlabel('signal median / signal background')
    plt.legend(loc='best')
    plt.show()

    if not os.path.exists('signal_over_background'): os.mkdir('signal_over_background')

    TFs = [f[:f.rfind('_HK.txt')] for f in HK_files]
    print(TFs)
    for i in range(len(HK_files)):
        plt.clf()
        me = me_data[i]
        hk = hk_data[i]
        plt.hist(me, bins=20, label=ME_files[i], alpha=0.5)
        plt.hist(hk, bins=20, label=HK_files[i], alpha=0.5)
        plt.xlabel('signal median / signal background')
        plt.legend(loc='best')
        plt.savefig('signal_over_background/{}_hist.png'.format(TFs[i]))

    for i in range(len(HK_files)):
        plt.clf()
        me = me_data[i]
        hk = hk_data[i]
        plt.plot(range(len(me)), me, 'o', label=ME_files[i])
        plt.plot(range(len(me), len(me)+len(hk)), hk, 'o', label=HK_files[i])
        plt.ylabel('signal median / signal background')
        plt.xlabel('probe')
        plt.legend(loc='best')
        plt.title(TFs[i])
        plt.savefig('signal_over_background/{}.png'.format(TFs[i]))




if __name__ == '__main__':
    main()
