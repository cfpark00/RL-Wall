#math_500
#python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/lorev_math_1_v1/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed" --lora_path "/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/ML/lorev/tests/math1/models/generation_4/lora_0"
#python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/lorev_math_1_v1.1/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed" --lora_path "/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/ML/lorev/tests/math1/models/generation_99/lora_5"
python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/lorev_math_1_v1.1/math_500/temp=0.4_n=16_ntokens=8192" --temperature 0.4 --n 16 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed" --lora_path "/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/ML/lorev/tests/math1/models/generation_99/lora_5"
#math_500
#printf "\n\n./data/new_evals_03_03/qwen-7b/lorev_math_1_v1/math_500/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/lorev_math_1_v1/math_500/temp=0.0_n=1_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/lorev_math_1_v1.1/math_500/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py "./data/new_evals_03_03/qwen-7b/lorev_math_1_v1.1/math_500/temp=0.0_n=1_ntokens=8192/data.json"
printf "\n\n./data/new_evals_03_03/qwen-7b/lorev_math_1_v1.1/math_500/temp=0.4_n=16_ntokens=8192/data.json\n"
python3 extract_correct.py "./data/new_evals_03_03/qwen-7b/lorev_math_1_v1.1/math_500/temp=0.4_n=16_ntokens=8192/data.json"
