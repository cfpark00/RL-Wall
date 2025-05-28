python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp\=1.0_n\=32_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_train/temp\=1.0_n\=32_ntokens\=8192/data.json --method verl
python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_500/temp\=1.0_n\=32_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_train/temp\=1.0_n\=32_ntokens\=8192/data.json --method verl