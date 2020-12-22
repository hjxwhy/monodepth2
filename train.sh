#!/bin/bash

GPUS='0,1,2,3,4,5,6,7'

NUM_GPUS=$(echo $GPUS | tr "," "\n" | wc -l)
NUM_THREADS=$(cat /proc/cpuinfo | grep processor | wc -l)

export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=$(($NUM_THREADS/$NUM_GPUS))

# launch_code() {
#     python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --node_rank=0 \
#     --master_port=2334 \
#     train.py \
#     --gpus $GPUS \
#     --png \
#     --load_weights_folder /root/tmp/mdp/models \
#     #  --batch_size 1 \
#     --model_name test_parallel
# }

python -m torch.distributed.launch \
 --nproc_per_node=$NUM_GPUS \
 --node_rank=0 \
 --master_port=2333 \
 train.py \
 --gpus $GPUS \
 --png
#  --model_name test_parallel
#  --load_weights_folder /root/tmp/mdp/models/weights_2