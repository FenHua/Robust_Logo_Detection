#!/usr/bin/env bash

CONFIG=/detect_config.py
CHECKPOINT=/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/checkpoints/epoch_21.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --format-only \
    --options "jsonfile_prefix=openBrand_result"
