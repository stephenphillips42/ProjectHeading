
py train.py \
  --dataset_dir=/scratch/synthscene/cubes128x128/ \
  --dataset_type=cubes128_depth \
  --output_type=invdepth \
  --network_type=hourglass \
  --batch_size=16 \
  --learning_rate=1e-4 \
  --weight_decay=0.0 \
  --weight_l1_decay=0.0 \
  --save_dir=/scratch/tfcheckpoints/depth_test03 \
  # End of args
