#!/bin/bash
set -e

# Overall settings
MEM=0512
MINIBATCH=0512
SEED=100
MDIR=`pwd`/../modlm
DEVEPOCHS=1

DTYPE=ptb
SENT_MULT=1

# DTYPE=ptbwsj
# SENT_MULT=10

# DTYPE=aspec0100
# SENT_MULT=2

# DTYPE=aspec0500
# SENT_MULT=10

# DTYPE=aspec2000
# SENT_MULT=40

SUF=$DTYPE
DATA=$DTYPE.train
VALID=$DTYPE.valid
TEST=$DTYPE.test

SETUP=
if [[ $SETUP == "dev" ]]; then
  DATAW=$DTYPE.train.all
  DATAA=$DTYPE.train.all
  HELDW=$DTYPE.valid
else
  DATAW=$DTYPE.train.WILD
  DATAA=$DTYPE.train.all
  HELDW=$DTYPE.held.WILD
fi
LANG=txt

# Set the models
for MY_DROPOUT_PROB in 0.5; do
for RECURRENCE in lstm; do
for MODID in mabsfixed; do
for MODID in mkn+one one mkn; do

BM="model-$SUF/$DATAA.uniform model-$SUF/$DATAA.unk"
DROPOUT_MODELS=""
HAS_ONE=no
HAS_DENSE=yes
if [[ $MODID == "mkn" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_4_3_2_1 $BM"
elif [[ $MODID == "mkn+wsj" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_4_3_2_1 model-wsj/wsj.train.ngram_mkn_4_3_2_1 $BM"
elif [[ $MODID == "wsj" ]]; then
  MODELS="model-wsj/wsj.train.ngram_mkn_4_3_2_1 $BM"
elif [[ $MODID == "mkn0+one" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn model-$SUF/$DATAA.onehot $BM"
  DROPOUT_MODELS="0"
  HAS_ONE=yes
elif [[ $MODID == "mkn1" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_1 $BM"
elif [[ $MODID == "mkn2" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_2_1 $BM"
elif [[ $MODID == "mkn3" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_3_2_1 $BM"
elif [[ $MODID == "mkns" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_4_3_2_1 model-$SUF/$DATAW.ngram_mkn_4_3_2 $BM"
elif [[ $MODID == "mknso" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_4_3_2 $BM"
elif [[ $MODID == "mabs" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mabs_4_3_2_1 $BM"
elif [[ $MODID == "mabsfixed" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mabsfixed_4_3_2_1 $BM"
elif [[ $MODID == "lin" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_lin_4_3_2_1 $BM"
elif [[ $MODID == "lin0+one" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_lin model-$SUF/$DATAW.onehot $BM"
  DROPOUT_MODELS="0"
  HAS_ONE=yes
elif [[ $MODID == "mkn+one" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_4_3_2_1 model-$SUF/$DATAA.onehot $BM"
  DROPOUT_MODELS="0"
  HAS_ONE=yes
elif [[ $MODID == "mkn+one+wsj" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_4_3_2_1 model-wsj/wsj.train.ngram_mkn_4_3_2_1 model-$SUF/$DATAA.onehot $BM"
  DROPOUT_MODELS="0,1"
  HAS_ONE=yes
elif [[ $MODID == "mkns+one" ]]; then
  MODELS="model-$SUF/$DATAW.ngram_mkn_4_3_2_1 model-$SUF/$DATAW.ngram_mkn_4_3_2 model-$SUF/$DATAA.onehot $BM"
  DROPOUT_MODELS="0,1"
  HAS_ONE=yes
elif [[ $MODID == "one" ]]; then
  MODELS="model-$SUF/$DATAA.onehot $BM"
  HAS_ONE=yes
  HAS_DENSE=no
else
  echo "Bad model id: $MODID"
  exit 1
fi

LOGID=$MODID

# Heuristic
HEURISTIC=
if [[ $HEURISTIC == "def" ]]; then
  if [[ $MODID = lin* ]]; then
    HEURISTIC=wb
  else
    HEURISTIC=abs
  fi
fi

# word history
WHIST=1
NODES=200
LAYERS=1
WREP=$NODES
if [[ $WHIST != 0 ]]; then
  LOGID="$LOGID+w${WHIST}x${WREP}"
fi

EVAL_FREQ=1
if [[ $HAS_ONE == "yes" ]]; then
  EVAL_FREQ=`echo "sqrt($NODES/50)*$SENT_MULT" | bc -l | sed 's/\..*//g'`
fi

# dropout
NODE_DROPOUT=0.5
if [[ $MODID == "mkn" ]] || [[ $MODID == "lin" ]] || [[ $MODID == "mabs" ]] || [[ $MODID == "mkns" ]] || [[ $MODID == "mknso" ]]; then NODE_DROPOUT=0.0; fi
DROPOUT_DECAY=1.0
DROPOUT_PROB=$MY_DROPOUT_PROB
if [[ $DROPOUT_MODELS == "" ]]; then DROPOUT_PROB=0.0; fi 
if [[ "$DROPOUT_PROB" != "0.0" ]]; then
  LOGID="$LOGID-do${DROPOUT_PROB}x${DROPOUT_DECAY}"
else
  DROPOUT_MODELS=""
fi 
if [[ "$NODE_DROPOUT" != "0.0" ]]; then
  LOGID="$LOGID-ndo${NODE_DROPOUT}"
fi

# trainer
TRAINING_TYPE=sent
TRAINER=adam
LR=0.001
RD=1.0
OEPOCHS=-1
CLIP="true"
if [[ $CLIP == "false" ]]; then CLIPID=nc; fi

if [[ $SETUP != "" ]]; then
  LOGID="$LOGID-$SETUP"
fi
if [[ $RD != "1.0" ]]; then
  LOGID="$LOGID-rd$RD"
fi

WHITEN=
if [[ $HAS_DENSE == "no" ]]; then WHITEN=; fi
WHITEN_EPS=0.001
if [[ $WHITEN != "" ]]; then
  LOGID="$LOGID-w$WHITEN"
  if [[ $WHITEN != "mean" ]]; then
    LOGID="$LOGID$WHITEN_EPS"
  fi
fi

EPOCHS=1000

# Directories
LOGID="$LOGID-$RECURRENCE${NODES}x$LAYERS-$TRAINING_TYPE-${TRAINER}${CLIPID}-mb$MINIBATCH-${LR}_o${OEPOCHS}-s$SEED"

if [[ $HEURISTIC != "" ]]; then
  LOGID="$MODID-heur$HEURISTIC";
  EPOCHS=1
  WHIST=0
  WHITEN=
  MINIBATCH=1
fi

mkdir -p result-$SUF
if [[ -e result-$SUF/$LOGID.log ]]; then
  echo "result-$SUF/$LOGID.log already exists"
else
  echo "result-$SUF/$LOGID.log running"
  {
    echo "$MDIR/src/modlm/modlm-train --dynet_seed $SEED --dynet_mem $MEM --wildcards \"all aa ab ac ad ae af ag ah ai aj\" --vocab_file model-$SUF/$DATA.vocab --train_file data-$SUF/$HELDW.$LANG --valid_file data-$SUF/$VALID.$LANG --test_file data-$SUF/$TEST.$LANG --dist_models \"$MODELS\" --learning_rate $LR --rate_decay $RD --model_dropout_prob $DROPOUT_PROB --model_dropout_decay $DROPOUT_DECAY --dropout_models \"$DROPOUT_MODELS\" --trainer $TRAINER --training_type $TRAINING_TYPE --clipping_enabled $CLIP --layers "$RECURRENCE:$NODES:$LAYERS" --node_dropout $NODE_DROPOUT --word_hist $WHIST --word_rep $WREP --penalize_unk false --epochs $EPOCHS --online_epochs $OEPOCHS --max_minibatch $MINIBATCH --whiten \"$WHITEN\" --whiten_eps $WHITEN_EPS --heuristic \"$HEURISTIC\" --evaluate_frequency $EVAL_FREQ --model_out result-$SUF/$LOGID.mod 2>&1 >> result-$SUF/$LOGID.log &"
    $MDIR/src/modlm/modlm-train --dynet_seed $SEED --dynet_mem $MEM --wildcards "all aa ab ac ad ae af ag ah ai aj" --vocab_file model-$SUF/$DATA.vocab --train_file data-$SUF/$HELDW.$LANG --valid_file data-$SUF/$VALID.$LANG --test_file data-$SUF/$TEST.$LANG --dist_models "$MODELS" --learning_rate $LR --rate_decay $RD --model_dropout_prob $DROPOUT_PROB --model_dropout_decay $DROPOUT_DECAY --dropout_models "$DROPOUT_MODELS" --training_type $TRAINING_TYPE --trainer $TRAINER --clipping_enabled $CLIP --layers "$RECURRENCE:$NODES:$LAYERS" --node_dropout $NODE_DROPOUT --word_hist $WHIST --word_rep $WREP --penalize_unk false --epochs $EPOCHS --online_epochs $OEPOCHS --max_minibatch $MINIBATCH --whiten "$WHITEN" --whiten_eps $WHITEN_EPS --heuristic "$HEURISTIC" --evaluate_frequency $EVAL_FREQ --model_out result-$SUF/$LOGID.mod 
  } &> result-$SUF/$LOGID.log &
fi

done
done
sleep 1
done
done
