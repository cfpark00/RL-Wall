#python3 eval.py --model_name qwen-2.5-1.5b-instruct --dataset_name math_500 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct/math_500/temp=1.0_seed=none" --temperature 1.0
#python3 eval.py --model_name qwen-2.5-1.5b-instruct --dataset_name aime_2024 --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct/aime_2024/temp=1.0_seed=none" --temperature 1.0

#python3 eval.py --model_name qwen-2.5-1.5b-instruct --dataset_name math_train --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct/math_train/temp=1.0_seed=none" --temperature 1.0 --resume


python3 eval.py --model_name qwen-2.5-1.5b-instruct --dataset_name math_train --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct/math_train/temp=0.6_seed=none" --temperature 0.6 --resume
python3 eval.py --model_name qwen-2.5-1.5b-instruct --dataset_name math_train --exp_dir "./data/baselines/qwen-2.5-1.5b-instruct/math_train/temp=0.2_seed=none" --temperature 0.2
