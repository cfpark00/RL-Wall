#python3 eval.py --model_name qwen-2.5-1.5b-instruct-grpo-math-v1-130 --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_500/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048
#python3 eval.py --model_name qwen-2.5-1.5b-instruct-grpo-math-v1-130 --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-grpo-math-v1-130/aime_2024/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048

#python3 eval.py --model_name qwen-2.5-1.5b-instruct-ppo-math-v1-130 --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-ppo-math-v1-130/math_500/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048
#python3 eval.py --model_name qwen-2.5-1.5b-instruct-ppo-math-v1-130 --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-ppo-math-v1-130/aime_2024/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048


#python3 eval.py --model_name qwen-2.5-1.5b-instruct-grpo-math-v1-130 --dataset_name math_train --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_train/temp=1.0_seed=none" --temperature 1.0  --max_tokens 2048

python3 eval.py --model_name qwen-2.5-1.5b-instruct-grpo-math-v1-130 --dataset_name math_train --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_train/temp=0.6_seed=none" --temperature 0.6  --max_tokens 2048
python3 eval.py --model_name qwen-2.5-1.5b-instruct-grpo-math-v1-130 --dataset_name math_train --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_train/temp=0.2_seed=none" --temperature 0.2  --max_tokens 2048
