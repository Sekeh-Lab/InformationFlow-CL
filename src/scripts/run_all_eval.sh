#!/bin/bash
### Script that calls for each task to be evaluated using the last saved checkpoint for that task
# Usage: ./scripts/run_all_eval.sh 0 vgg16 0.8,0.8,0.8,0.8,0.8,-1 run0


GPU_ID=$1
ARCH=$2
PRUNE=$3
RUN_ID=$4

TAG='nobias-nobn'
EXTRA_FLAGS=''

echo $ARCH'-'$TAG'_'$RUN_ID

bash ./scripts/run_eval.sh $GPU_ID $ARCH $PRUNE \
   $ARCH'-'$TAG'_'$RUN_ID #$EXTRA_FLAGS

