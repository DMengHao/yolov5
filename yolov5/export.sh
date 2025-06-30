#! /bin/bash
python ./export.py --data ./data/dk01.yaml --weights ./weights/crossing01.pt --include onnx engine --batch-size 1 --inplace --half --imgsz 1280