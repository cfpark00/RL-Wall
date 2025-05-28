python3 force_answer.py --model_name qwen-2.5-0.5b-instruct --exp_dir "./data/new_evals_03_03/qwen-2.5-0.5b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --thresholds "512,724,1024,1448,2048,2896,4096,5792,8192" --tensor_parallel_size 2
#python3 force_answer.py --model_name qwen-2.5-1.5b-instruct --exp_dir "./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --thresholds "512,724,1024,1448,2048,2896,4096,5792,8192" --tensor_parallel_size 2
#python3 force_answer.py --model_name qwen-2.5-3b-instruct --exp_dir "./data/new_evals_03_03/qwen-2.5-3b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --thresholds "512,724,1024,1448,2048,2896,4096,5792,8192" --tensor_parallel_size 4
python3 force_answer.py --model_name qwen-2.5-7b-instruct --exp_dir "./data/new_evals_03_03/qwen-2.5-7b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --thresholds "512,724,1024,1448,2048,2896,4096,5792,8192" --tensor_parallel_size 4
python3 force_answer.py --model_name qwen-2.5-14b-instruct --exp_dir "./data/new_evals_03_03/qwen-2.5-14b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --thresholds "512,724,1024,1448,2048,2896,4096,5792,8192" --tensor_parallel_size 4
python3 force_answer.py --model_name qwen-2.5-32b-instruct --exp_dir "./data/new_evals_03_03/qwen-2.5-32b-instruct/math_500/temp=1.0_n=8_ntokens=8192" --thresholds "512,724,1024,1448,2048,2896,4096,5792,8192" --tensor_parallel_size 4
python3 force_answer.py --model_name s1-32b --exp_dir "./data/new_evals_03_03/s1-32b/math_500/temp=1.0_n=8_ntokens=8192" --thresholds "512,724,1024,1448,2048,2896,4096,5792,8192" --tensor_parallel_size 4



