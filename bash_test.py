
import os


cmd = "/home/tanyi/miniconda3/envs/videonet/bin/python test_models.py somethingv1 RGB \
     --arch resnet50 --test_nets Afcstlfq --test_segments 8 \
     --element_filter --cdiv 4 \
     --batch-size 8 -j 8 --dropout 0.5 --consensus_type=avg \
     --test_crops 1 \
     --test_weights=checkpoint/C2D_Afcstlfq_resnet50_a4b2_somethingv1_RGB_avg_segment8_e50_s20p40_twice_ef4_drp5_lr01_b36_npb_190321h14/ckpt_test.best.pth.tar"

os.system(cmd)
