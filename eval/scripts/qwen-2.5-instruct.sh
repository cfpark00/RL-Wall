python3 generate_responses.py --model_name qwen-2.5-0.5b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-2.5-0.5b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --temperature 1.0 --n 8 --max_tokens 8192 --tensor_parallel_size 2
python3 generate_responses.py --model_name qwen-2.5-1.5b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --temperature 1.0 --n 8 --max_tokens 8192 --tensor_parallel_size 4
python3 generate_responses.py --model_name qwen-2.5-3b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-2.5-3b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --temperature 1.0 --n 8 --max_tokens 8192 --tensor_parallel_size 4
python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-2.5-7b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --temperature 1.0 --n 8 --max_tokens 8192 --tensor_parallel_size 4
python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-2.5-14b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --temperature 1.0 --n 8 --max_tokens 8192 --tensor_parallel_size 4
python3 generate_responses.py --model_name qwen-2.5-32b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-2.5-32b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --temperature 1.0 --n 8 --max_tokens 8192 --tensor_parallel_size 4
python3 generate_responses.py --model_name s1-32b --dataset_name math_500 --exp_dir "./data/new_evals_03_03/s1-32b/math_500/temp=1.0_n=8_ntokens=8192" --temperature 1.0 --n 8 --max_tokens 8192 --tensor_parallel_size 4



