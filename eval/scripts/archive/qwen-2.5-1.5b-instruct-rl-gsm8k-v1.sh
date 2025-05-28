#python3 eval.py --model_name qwen-2.5-1.5b-instruct-grpo-gsm8k-v1-7 --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-grpo-gsm8k-v1-7/math_500/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048
#python3 eval.py --model_name qwen-2.5-1.5b-instruct-grpo-gsm8k-v1-7 --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-grpo-gsm8k-v1-7/aime_2024/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048

#python3 eval.py --model_name qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7 --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7/math_500/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048
#python3 eval.py --model_name qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7 --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7/aime_2024/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048
python3 eval.py --model_name qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7 --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7/math_500/temp=0.7_seed=none" --temperature 0.7  --max_tokens 2048
python3 eval.py --model_name qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7 --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7/aime_2024/temp=0.7_seed=none" --temperature 0.7  --max_tokens 2048
