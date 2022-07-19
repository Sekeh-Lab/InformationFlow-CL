#!/bin/bash
# Script that calls function to add tasks in sequence using the iterative 
# pruning + re-training method.
# Usage:
# ./scripts/run_all_sequence.sh 3 vgg16 1

GPU_ID=$1
ARCH=$2
PRUNE=$3
FREEZE_PERC=$4
NUM_FREEZE_LAYERS=$5
FREEZE_ORDER=$6
RUN_ID=$7
TRAIN_BN=$8

TAG='nobias-nobn'
EXTRA_FLAGS=''

echo $ARCH'-'$TAG'_'$RUN_ID

bash ./scripts/run_sequence.sh $GPU_ID $ARCH $PRUNE $FREEZE_PERC $NUM_FREEZE_LAYERS \
   $FREEZE_ORDER $ARCH'-'$TAG'_'$RUN_ID $TRAIN_BN
  #$EXTRA_FLAGS

