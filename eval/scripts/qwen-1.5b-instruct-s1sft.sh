#math_500
python3 generate_responses.py --model_name qwen-2.5-1.5b-instruct-s1sft-v1-200 --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-1.5b/qwen-2.5-1.5b-instruct-s1sft-v1-200/math_500/temp=0.0_n=1_ntokens=16384" --temperature 0.0 --n 1 --max_tokens 16384 --tensor_parallel_size 4 --prompt_suffix "_boxed"
printf "\n\n./data/new_evals_03_03/qwen-1.5b/qwen-2.5-1.5b-instruct-s1sft-v1-200/math_500/temp=0.0_n=1_ntokens=16384/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-1.5b/qwen-2.5-1.5b-instruct-s1sft-v1-200/math_500/temp=0.0_n=1_ntokens=16384/data.json

python3 generate_responses.py --model_name qwen-2.5-1.5b-instruct-s1sft-v1-200 --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-1.5b/qwen-2.5-1.5b-instruct-s1sft-v1-200/math_500/temp=0.4_n=64_ntokens=16384" --temperature 0.4 --n 64 --max_tokens 16384  --tensor_parallel_size 4 --prompt_suffix "_boxed"
printf "\n\n./data/new_evals_03_03/qwen-1.5b/qwen-2.5-1.5b-instruct-s1sft-v1-200/math_500/temp=0.4_n=64_ntokens=16384/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-1.5b/qwen-2.5-1.5b-instruct-s1sft-v1-200/math_500/temp=0.4_n=64_ntokens=16384/data.json