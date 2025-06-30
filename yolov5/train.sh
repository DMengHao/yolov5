#! /bin/bash
python -m torch.distributed.run --nproc_per_node 4 train.py --batch-size 128 --epochs 100 --project ./runs_dk01/train --data ./data/dk01.yaml --device 0,1,2,3 --cache disk