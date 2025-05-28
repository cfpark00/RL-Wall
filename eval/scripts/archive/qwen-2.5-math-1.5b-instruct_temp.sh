#0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 1.3, 1.6
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=0.003_seed=none" --temperature 0.003 --max_tokens 2048
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=0.01_seed=none" --temperature 0.01 --max_tokens 2048
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=0.03_seed=none" --temperature 0.03 --max_tokens 2048
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=0.1_seed=none" --temperature 0.1 --max_tokens 2048
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=0.3_seed=none" --temperature 0.3 --max_tokens 2048
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=1.0_seed=none" --temperature 1.0 --max_tokens 2048
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=1.3_seed=none" --temperature 1.3 --max_tokens 2048
python3 eval.py --model_name qwen-2.5-math-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-math-1.5b-instruct/math_500/temp_scale/temp=1.6_seed=none" --temperature 1.6 --max_tokens 2048

