python3 generate_responses.py --model_name qwen-2.5-3b-instruct --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-3b/qwen-2.5-3b-instruct/math_train/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
python3 generate_responses.py --model_name qwen-3b-grpo-math-200steps --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-3b/qwen-3b-grpo-math-200steps/math_train/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#python3 generate_responses.py --model_name qwen-2.5-3b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-3b/qwen-2.5-3b-instruct/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwen-3b-grpo-math-200steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-3b/qwen-3b-grpo-math-200steps/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"


#python3 generate_responses.py --model_name qwen-2.5-3b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-3b/qwen-2.5-3b-instruct/math_500/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
python3 generate_responses.py --model_name qwen-3b-grpo-math-200steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-3b/qwen-3b-grpo-math-200steps/math_500/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
