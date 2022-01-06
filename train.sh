#!/bin/bash

python train.py --data_path /public/datasets_neo/airsim_uav \
  --split sim \
  --dataset uav \
  --png \
  --height 480 --width 640 \
  --batch_size 7 \
  --log_dir /home/hjx/workspace/mdp \
  --num_workers 12 \
  --semi_sup \
  --use_sparse \
  --max_depth 200 \
  --model_name sparse_weight_1 \
  --num_epochs 50
# --sparse_weight 0.5
# change model name should also change eval weight folder name
# the max depth should be the same
python evaluate_depth_scale.py \
  --data_path /public/datasets_neo/airsim_uav--eval_split sim \
  --dataset uav \
  --png \
  --height 480 --width 640 \
  --batch_size 7 \
  --log_dir /home/hjx/workspace/mdp \
  --num_workers 12 \
  --max_depth 200 \
  --load_weights_folder /home/hjx/workspace/mdp/sparse_weight_1/models/weights_49

python train.py --data_path /public/datasets_neo/airsim_uav \
  --split sim \
  --dataset uav \
  --png \
  --height 480 --width 640 \
  --batch_size 7 \
  --log_dir /home/hjx/workspace/mdp \
  --num_workers 12 \
  --semi_sup \
  --use_sparse \
  --max_depth 200 \
  --model_name sparse_weight_05 \
  --num_epochs 50 \
  --sparse_weight 0.5

python evaluate_depth_scale.py \
  --data_path /public/datasets_neo/airsim_uav--eval_split sim \
  --dataset uav \
  --png \
  --height 480 --width 640 \
  --batch_size 7 \
  --log_dir /home/hjx/workspace/mdp \
  --num_workers 12 \
  --max_depth 200 \
  --load_weights_folder /home/hjx/workspace/mdp/sparse_weight_05/models/weights_49
#GPUS='0'
#if [[ ! -z $TEST && $TEST = true ]] ; then
#    ARGS="--model_name test_parallel --num_workers 1"
#else
#    ARGS="--num_workers 20 "
#fi
#if [ -z $PORT ] ; then
#    PORT=$(($RANDOM%8192+8192))
#fi
#if [ -z $CUDA_VISIBLE_DEVICES ] ; then
#    export CUDA_VISIBLE_DEVICES=$GPUS
#fi
#
#NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
## NUM_THREADS=$(cat /proc/cpuinfo | grep processor | wc -l)
#echo "using GPU ["$CUDA_VISIBLE_DEVICES"] on port: "$PORT
#
## export OMP_NUM_THREADS=$(($NUM_THREADS/$NUM_GPUS))
#export OMP_NUM_THREADS=16
#
#python -m torch.distributed.launch \
# --nproc_per_node=$NUM_GPUS \
# --node_rank=0 \
# --master_port=$PORT \
# train.py \
# --png $ARGS $@
#
#kill -9 $(ps aux | grep '='$PORT | grep -v grep | awk '{print $2}')
