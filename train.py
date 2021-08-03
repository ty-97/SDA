import os


cmd = "/home/tanyi/miniconda3/envs/videonet/bin/python main.py somethingv2 RGB \
     --model TSN --arch resnet50 --net SDA_full --num_segments 8 \
     --cdiv 4 \
     --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
     --batch-size 36 -j 4 --dropout 0.5\
     --npb --twice_sample "

os.system(cmd)
