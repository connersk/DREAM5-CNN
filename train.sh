set -e
set -x

OUTDIR=$1

for i in $(cat data/train_TFs.txt)
do
    echo $i
    python test_convnet.py data/${i}_ME.txt data/${i}_HK.txt -o ${OUTDIR}/${i}_trainME.pkl > ${OUTDIR}/${i}_trainME.log
    python test_convnet.py data/${i}_HK.txt data/${i}_ME.txt -o ${OUTDIR}/${i}_trainHK.pkl > ${OUTDIR}/${i}_trainHK.log
    python test_convnet.py data/${i}_ME.txt data/${i}_HK.txt -o ${OUTDIR}_rc/${i}_trainME.pkl --incl-rev-comp > ${OUTDIR}_rc/${i}_trainME.log
    python test_convnet.py data/${i}_HK.txt data/${i}_ME.txt -o ${OUTDIR}_rc/${i}_trainHK.pkl --incl-rev-comp > ${OUTDIR}_rc/${i}_trainHK.log
done
