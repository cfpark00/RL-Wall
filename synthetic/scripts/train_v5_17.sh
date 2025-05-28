#!/bin/bash
echo ""
echo ./configs/v5_16.yaml
#accelerate launch
python3 /n/home12/cfpark00/ML/tools/run_sft_accelerate.py ./configs/v5_17.yaml --overwrite

