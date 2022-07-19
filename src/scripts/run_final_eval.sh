#!/bin/bash
# Adds tasks in sequence using the iterative pruning + re-training method.
# Usage:
# ./scripts/run_sequence.sh ORDER PRUNE_STR LOADNAME GPU_IDS RUN_TAG EXTRA_FLAGS
# ./scripts/run_sequence.sh csf 0.75,0.75,-1 ../checkpoints/imagenet/imagenet_pruned_0.5_final.pt 3 nobias_1 

# This is hard-coded to prevent silly mistakes.
declare -A DATASETS
# DATASETS=["CIFAR100", "CIFAR101", "CIFAR102", "CIFAR103", "CIFAR104"]
DATASETS[0]="CIFAR10"
DATASETS[1]="CIFAR100"
DATASETS[2]="CIFAR101"
DATASETS[3]="CIFAR102"
DATASETS[4]="CIFAR103"
DATASETS[5]="CIFAR104"



GPU_IDS=$1     #$GPU__ID
ARCH=$2
PRUNE_STR=$3   #0.75,0.75,-1
RUN_TAG=$4     #VGG16_0.5-nobias-nobn_1 
EXTRA_FLAGS=$5 # --train_biases or --train_bn
 
  
# Prepare tags and savenames.
### Tag just denotes the order of tasks and their pruning amounts for file names
for (( i=0; i<6; i++)); do
  dataset=${DATASETS[$i]}
  if [ $i -eq 5 ]
    then
        preprune=1
    else
        echo 'Postprune'
        preprune=0
  fi
  # Get model to add dataset to.
  loadname=../checkpoints/CIFAR104/$PRUNE_STR/$RUN_TAG'_pruned_final.pt'

  ### This is "if ($loadname doesn't exist)"
  if [ ! -f $loadname ]; 
    then
      echo 'Final file not found! Using postprune'
      loadname=../checkpoints/CIFAR104/$PRUNE_STR/$RUN_TAG'_pruned_postprune.pt'
      if [ ! -f $loadname ]; 
        then
          echo 'Postprune file not found! Using preprune'
          loadname=../checkpoints/CIFAR104/$PRUNE_STR/$RUN_TAG'.pt'
      fi
  fi



  ### Finetuning file name, runtag is arbitrary run #
  logname=../logs/CIFAR104/$PRUNE_STR/$RUN_TAG

  ##############################################################################
  # Evaluate on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode eval $EXTRA_FLAGS \
    --dataset $dataset --loadname $loadname --preprune $preprune --lr 1e-4 --lr_decay_every 10 \
     --lr_decay_factor 0.1 | tee $logname'_eval.txt'

  # prev_pruned_savename=$pruned_savename
done
