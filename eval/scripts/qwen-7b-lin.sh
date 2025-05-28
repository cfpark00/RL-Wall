#
#python3 generate_responses.py --model_name qwen-2.5-7b-instruct-lin-v1 --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct-lin-v1/math_500/temp=0.0_n=1_ntokens=32768" --temperature 0.0 --n 1 --max_tokens 32768 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwen-2.5-7b-instruct-lin-v1 --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct-lin-v1/math_500/temp=0.4_n=8_ntokens=32768" --temperature 0.4 --n 8 --max_tokens 32768 --tensor_parallel_size 2 --prompt_suffix "_boxed"
python3 generate_responses.py --model_name qwen-2.5-7b-instruct-lin-v2-1 --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct-lin-v2-1/math_500/temp=0.0_n=1_ntokens=32768" --temperature 0.0 --n 1 --max_tokens 32768 --tensor_parallel_size 2 --prompt_suffix "_boxed"
python3 generate_responses.py --model_name qwen-2.5-7b-instruct-lin-v2-2 --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct-lin-v2-2/math_500/temp=0.0_n=1_ntokens=32768" --temperature 0.0 --n 1 --max_tokens 32768 --tensor_parallel_size 2 --prompt_suffix "_boxed"
python3 generate_responses.py --model_name qwen-2.5-7b-instruct-lin-v2-3 --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct-lin-v2-3/math_500/temp=0.0_n=1_ntokens=32768" --temperature 0.0 --n 1 --max_tokens 32768 --tensor_parallel_size 2 --prompt_suffix "_boxed"



