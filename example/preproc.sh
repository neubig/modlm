#!/bin/bash
set -e
set -o pipefail

MDIR=`pwd`/..

# SUF=wsj
# LANG=txt
# DATA=wsj.train
# SIZE=

SUF=ptb
LANG=txt
DATA=ptb.train
SIZE=4207

# SUF=aspec0100
# LANG=txt
# DATA=aspec0100.train
# SIZE=10000

# SUF=aspec0500
# LANG=txt
# DATA=aspec0500.train
# SIZE=50000

# SUF=aspec2000
# LANG=txt
# DATA=aspec2000.train
# SIZE=200000

# SUF=ptbwsj
# LANG=txt
# DATA=ptbwsj.train
# SIZE=186000

mkdir -p model-$SUF
cat data-$SUF/$SUF.train.txt | ./make-vocab.pl > model-$SUF/$SUF.train.vocab

split -l $SIZE data-$SUF/$SUF.train.txt data-$SUF/$SUF.held.
for f in data-$SUF/$SUF.held.a[abcdefghij]; do
  mv $f $f.txt
done
for f in all aa ab ac ad ae af ag ah ai aj; do
  [[ -e data-$SUF/$SUF.train.$f.txt ]] && rm data-$SUF/$SUF.train.$f.txt
  for g in aa ab ac ad ae af ag ah ai aj; do
    if [[ $f != $g ]]; then
      cat data-$SUF/$SUF.held.$g.txt >> data-$SUF/$SUF.train.$f.txt
    fi
  done
done

for DATA in $SUF.train.{all,aa,ab,ac,ad,ae,af,ag,ah,ai,aj}; do

{

mkdir -p model-$SUF
for sig in ngram_mkn_4_3_2_1 onehot unk uniform; do
  echo "$MDIR/src/modlm/dist-train --sig $sig --train_file data-$SUF/$DATA.$LANG --model_out model-$SUF/$DATA.$sig"
  time $MDIR/src/modlm/dist-train --sig $sig --train_file data-$SUF/$DATA.$LANG --model_out model-$SUF/$DATA.$sig
  gzip -f model-$SUF/$DATA.$sig
done

} &

if [[ $DATA == "$SUF.train.ab" ]]; then wait; fi
if [[ $DATA == "$SUF.train.ae" ]]; then wait; fi
if [[ $DATA == "$SUF.train.ah" ]]; then wait; fi

done

wait


