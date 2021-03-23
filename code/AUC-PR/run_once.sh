#!/usr/bin/env bash


DATASET=$1
MR=$2
NUM_NEG=$3
HOP=$4
LR=$5
L2=$6
EXP_NAME=$7
EPOCH=$8
GPU=$9


python train.py -d "$DATASET" -e "$DATASET"_"$EXP_NAME"_"$EPOCH" --num_epochs $EPOCH -g $GPU --hop $HOP --lr $LR --margin $MR \
--num_neg_samples_per_link $NUM_NEG --l2 $L2 --num_gcn_layers 2 --no_jk
python test_auc.py -d "$DATASET"_"ind" -e "$DATASET"_"$EXP_NAME"_"$EPOCH" -g $GPU --hop $HOP --runs 5
