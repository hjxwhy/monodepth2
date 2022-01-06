python evaluate_depth_scale.py \
  --data_path /public/datasets_neo/airsim_uav--eval_split sim \
  --dataset uav --png \
  --height 480 --width 640 \
  --batch_size 7 \
  --log_dir /home/hjx/workspace/mdp \
  --num_workers 12 \
  --max_depth 200 \
  --load_weights_folder /home/hjx/workspace/mdp/tmp/models/weights_19
