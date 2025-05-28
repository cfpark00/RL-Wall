#python3 eval.py --model_name qwen-2.5-32b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-32b-instruct/math_500/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048
#python3 eval.py --model_name qwen-2.5-32b-instruct --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-32b-instruct/aime_2024/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048

#python3 eval.py --model_name qwen-2.5-32b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-32b-instruct/math_500/temp=0.6_seed=none" --temperature 0.6  --max_tokens 2048
#python3 eval.py --model_name qwen-2.5-32b-instruct --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-32b-instruct/aime_2024/temp=0.6_seed=none" --temperature 0.6  --max_tokens 2048

#python3 eval.py --model_name qwen-2.5-32b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-32b-instruct/math_500/temp=0.2_seed=none" --temperature 0.2  --max_tokens 2048
#python3 eval.py --model_name qwen-2.5-32b-instruct --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-32b-instruct/aime_2024/temp=0.2_seed=none" --temperature 0.2  --max_tokens 2048


python3 eval.py --model_name qwen-2.5-32b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-32b-instruct/math_500/temp=0.2_seed=none_n=1_ntokens=16384" --temperature 1.0  --max_tokens 16384 --n 1


