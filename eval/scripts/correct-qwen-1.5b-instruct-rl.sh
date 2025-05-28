#baseline
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp=1.0_n=32_ntokens=8192/data.json
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp=1.0_n=128_ntokens=8192/data.json
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp=0.0_n=1_ntokens=8192/data.json
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_train/temp=1.0_n=32_ntokens=8192/data.json
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_train/temp=0.0_n=1_ntokens=8192/data.json
#gsm8k
python3 extract_correct.py ./data/new_evals_03_03/qwen-1.5b/qwen-2.5-1.5b-instruct/gsm8k_train/temp=1.0_n=16_ntokens=8192/data.json

#post rl
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_500/temp=1.0_n=32_ntokens=8192/data.json
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_500/temp=0.0_n=1_ntokens=8192/data.json
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_train/temp=1.0_n=32_ntokens=8192/data.json
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_train/temp=0.0_n=1_ntokens=8192/data.json
