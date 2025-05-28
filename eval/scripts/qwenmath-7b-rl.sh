#math_train
##python3 generate_responses.py --model_name qwen-2.5-math-7b-instruct --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwen-2.5-math-7b-instruct/math_train/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwenmath-simplerl --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwenmath-simplerl/math_train/temp=0._n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_train t=0.6
##python3 generate_responses.py --model_name qwen-2.5-math-7b-instruct --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwen-2.5-math-7b-instruct/math_train/temp=0.6_n=16_ntokens=8192" --temperature 0.6 --n 16 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwenmath-simplerl --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwenmath-simplerl/math_train/temp=0.6_n=16_ntokens=8192" --temperature 0.6 --n 16 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"


#math_500
##python3 generate_responses.py --model_name qwen-2.5-math-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwen-2.5-math-7b-instruct/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwenmath-simplerl --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwenmath-simplerl/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=0.6
##python3 generate_responses.py --model_name qwen-2.5-math-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwen-2.5-math-7b-instruct/math_500/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwenmath-simplerl --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwenmath-simplerl/math_500/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=1.0
##python3 generate_responses.py --model_name qwen-2.5-math-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwen-2.5-math-7b-instruct/math_500/temp=1.0_n=64_ntokens=8192" --temperature 1.0 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwenmath-simplerl --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwenmath-simplerl/math_500/temp=1.0_n=64_ntokens=8192" --temperature 1.0 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#aime_2024
python3 generate_responses.py --model_name qwen-2.5-math-7b-instruct --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwen-2.5-math-7b-instruct/aime_2024/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwenmath-simplerl --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwenmath-simplerl/aime_2024/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#aime_2024 t=0.6
python3 generate_responses.py --model_name qwen-2.5-math-7b-instruct --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwen-2.5-math-7b-instruct/aime_2024/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwenmath-simplerl --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwenmath-7b/qwenmath-simplerl/aime_2024/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"



