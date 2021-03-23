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

for idx in "0" "1" "2" "3" "4"
do
    bash best_once.sh $DATASET $MR $NUM_NEG $HOP $LR $L2 "$EXP_NAME"_$idx $EPOCH $GPU
done
