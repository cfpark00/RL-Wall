python3 generate_responses.py --model_name simplerl-re-7b-40steps --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/simplerl/simplerl-re-7b-40steps/aime_2024/temp=0.0_n=1_ntokens=16384" --temperature 0.0 --n 1 --max_tokens 16384 --tensor_parallel_size 2 --prompt_suffix "_boxed"
python3 generate_responses.py --model_name simplerl-re-7b-100steps --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/simplerl/simplerl-re-7b-100steps/aime_2024/temp=0.0_n=1_ntokens=16384" --temperature 0.0 --n 1 --max_tokens 16384 --tensor_parallel_size 2 --prompt_suffix "_boxed"

python3 generate_responses.py --model_name simplerl-re-7b-40steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/simplerl/simplerl-re-7b-40steps/math_500/temp=0.0_n=1_ntokens=16384" --temperature 0.0 --n 1 --max_tokens 16384 --tensor_parallel_size 2 --prompt_suffix "_boxed"
python3 generate_responses.py --model_name simplerl-re-7b-100steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/simplerl/simplerl-re-7b-100steps/math_500/temp=0.0_n=1_ntokens=16384" --temperature 0.0 --n 1 --max_tokens 16384 --tensor_parallel_size 2 --prompt_suffix "_boxed"

