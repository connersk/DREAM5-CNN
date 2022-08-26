set -e
set -x

OUTDIR=$1

for i in $(cat data/train_TFs.txt)
do
    echo $i
    python classification_convnet.py data/${i}_ME.txt data/${i}_HK.txt -o ${OUTDIR}_single_over/${i}_trainME.pkl --incl-rev-comp  --one_array --sampling oversampling --sampling_factor 10 > ${OUTDIR}_single_over/${i}_trainME.log
    python classification_convnet.py data/${i}_HK.txt data/${i}_ME.txt -o ${OUTDIR}_single_over/${i}_trainHK.pkl --incl-rev-comp  --one_array --sampling oversampling --sampling_factor 10 > ${OUTDIR}_single_over/${i}_trainHK.log
done
