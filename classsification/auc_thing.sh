set -e
set -x

OUTDIR=$1

for i in $(cat data/train_TFs.txt)
do
    echo $i
    python classification/roc_plt.py class_single_over_opt/${i}_trainHK.pkl
    python classification/roc_plt.py class_single_over_opt/${i}_trainMe.pkl
    
done
