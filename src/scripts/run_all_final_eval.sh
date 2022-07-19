#!/bin/bash
### Script that calls for evaluation of all tasks using the last saved checkpoint of the last task
###    by reapplying the stored task-specific binary masks to ensure there is no forgetting.
# Usage: ./scripts/run_all_final_eval.sh 0 vgg16 0.8,0.8,0.8,0.8,0.8,-1 run0


GPU_ID=$1
ARCH=$2
PRUNE=$3
RUN_ID=$4

TAG='nobias-nobn'
EXTRA_FLAGS=''

echo $ARCH'-'$TAG'_'$RUN_ID

bash ./scripts/run_final_eval.sh $GPU_ID $ARCH $PRUNE \
   $ARCH'-'$TAG'_'$RUN_ID #$EXTRA_FLAGS

