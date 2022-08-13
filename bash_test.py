
import os


cmd = "/home/tanyi/.conda/envs/vidnet/bin/python test_models.py somethingv1 RGB \
     --model TSN --arch resnet50 --test_nets SDA --test_segments 8 \
     --cdiv 4 \
     --batch-size 8 -j 8 --dropout 0.5 --consensus_type=avg \
     --full_res --test_crops 1\
     --test_weights="

os.system(cmd)
