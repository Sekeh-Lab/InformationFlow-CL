#!/bin/bash
# Runs all six tasks. Initially loads creates and loads the models starting checkpoint with a calls to --init_dump, and subsequently 
#    repeats a loop of training, calculating connectivities, and pruning for each task in sequence. 
# Usage:
# This script is called through the use of all_sequences.sh, see it for usage details

# This is hard-coded to prevent silly mistakes.
declare -A DATASETS
DATASETS[0]="CIFAR10"
DATASETS[1]="CIFAR100"
DATASETS[2]="CIFAR101"
DATASETS[3]="CIFAR102"
DATASETS[4]="CIFAR103"
DATASETS[5]="CIFAR104"



GPU_ID=$1
ARCH=$2
PRUNE_STR=$3               #0.75,0.75,-1
FREEZE_PERC=$4
NUM_FREEZE_LAYERS=$5
FREEZE_ORDER=$6
RUN_TAG=$7     #VGG16_0.5-nobias-nobn_1 
TRAIN_BN=$8
EXTRA_FLAGS='' # --train_biases or --train_bn


echo $RUN_TAG
echo $PRUNE

### Split prune string by the commas
IFS=',' read -r -a PRUNE <<< $PRUNE_STR

echo $PRUNE

for (( i=0; i<6; i++ )); do
  dataset=${DATASETS[$i]}
  prune=${PRUNE[$i]}
  ### Default number of classes is 20, for the CIFAR-100 tasks (2-6). Changes to 1 for CIFAR-10
  numoutputs=20
  if [ $i -eq 0 ]
    then
      numoutputs=10
  fi
  # Get model to add dataset to.
  if [ $i -eq 0 ]
    then
      CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --arch $ARCH --init_dump
      loadname=../checkpoints/$dataset/$ARCH.pt
    else
      loadname=$prev_pruned_savename'_final.pt'

      ### This is "if ($loadname doesn't exist)"
      if [ ! -f $loadname ]; then
          echo 'Final file not found! Using postprune'
          loadname=$prev_pruned_savename'_postprune.pt'
      fi
  fi

  if [ ! -d ../checkpoints/$dataset/$PRUNE_STR/ ]; 
    then
      echo 'Making Prune Directory'
      mkdir ../checkpoints/$dataset/$PRUNE_STR/
  fi

  
    if [ ! -d ../logs/$dataset/$PRUNE_STR/ ]; 
    then
      echo 'Making Prune Directory'
      mkdir ../logs/$dataset/$PRUNE_STR/
  fi
  
  # Prepare savenames.
  ### Finetuning file name, runtag is arbitrary run #
  ft_savename=../checkpoints/$dataset/$PRUNE_STR/$RUN_TAG
  conn_savename=../checkpoints/$dataset/$PRUNE_STR/$RUN_TAG'_conns'
  pruned_savename=../checkpoints/$dataset/$PRUNE_STR/$RUN_TAG'_pruned'
  logname=../logs/$dataset/$PRUNE_STR/$RUN_TAG

  ### If training stops partway you can skip to the correct task by changing [$i -lt <tasknum>] accordingly
  if [ $i -lt 0 ]
    then 
      echo 'skipping'
    else
      ##############################################################################
      # Train on current dataset.
      ##############################################################################
      CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode finetune $EXTRA_FLAGS \
        --dataset $dataset --num_outputs $numoutputs \
        --loadname $loadname --task_num $i \
        --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 --finetune_epochs 20 --train_bn $TRAIN_BN  \
        --save_prefix $ft_savename | tee $logname'.txt'
      


      ##############################################################################
      # Compute connectivity
      ##############################################################################
      CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode conns $EXTRA_FLAGS \
      --dataset $dataset --num_outputs $numoutputs \
      --loadname $ft_savename'.pt' --task_num $i --save_conn_name $conn_savename \
      --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 --finetune_epochs 20 --train_bn $TRAIN_BN \
      --save_prefix $ft_savename | tee $logname'.txt'
      
   
      echo $prune
      ##############################################################################
      # Prune on current dataset.
      ##############################################################################
      CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode prune $EXTRA_FLAGS \
        --dataset $dataset --loadname $ft_savename'.pt' --task_num $i --freeze_order $FREEZE_ORDER \
        --prune_perc_per_layer $prune --post_prune_epochs 10 --freeze_perc $FREEZE_PERC --num_freeze_layers $NUM_FREEZE_LAYERS \
        --lr 1e-4 --lr_decay_every 10 --lr_decay_factor 0.1 --train_bn $TRAIN_BN --num_outputs $numoutputs \
        --save_prefix $pruned_savename | tee $logname'_pruned.txt'
  fi
  prev_pruned_savename=$pruned_savename
done





