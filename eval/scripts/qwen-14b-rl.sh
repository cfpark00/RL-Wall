#math_train
#python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_train/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_train/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_train t=0.6
#python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_train/temp=0.6_n=16_ntokens=8192" --temperature 0.6 --n 16 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_train/temp=0.6_n=16_ntokens=8192" --temperature 0.6 --n 16 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

###
#math_500
python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=0.4
python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=0.6
#python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=1.0
#python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=1.0_n=64_ntokens=8192" --temperature 1.0 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=1.0_n=64_ntokens=8192" --temperature 1.0 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

###
#aime_2024
python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#aime_2024 t=0.4
python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#aime_2024 t=0.6
#python3 generate_responses.py --model_name qwen-2.5-14b-instruct --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name QWEN14BRL --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"



##correct
#math_train
#printf "\n\n./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_train/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_train/temp=0.0_n=1_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_train/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_train/temp=0.0_n=1_ntokens=8192/data.json

#printf "\n\n./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_train/temp=0.6_n=16_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_train/temp=0.6_n=16_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_train/temp=0.6_n=16_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_train/temp=0.6_n=16_ntokens=8192/data.json

#math_500
printf "\n\n./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.0_n=1_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.0_n=1_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.0_n=1_ntokens=8192/data.json

printf "\n\n/data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.4_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.4_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.4_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.4_n=64_ntokens=8192/data.json

printf "\n\n/data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.6_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=0.6_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.6_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=0.6_n=64_ntokens=8192/data.json

printf "\n\n./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=1.0_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/math_500/temp=1.0_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=1.0_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/math_500/temp=1.0_n=64_ntokens=8192/data.json

#aime_2024
printf "\n\n./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.0_n=1_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.0_n=1_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.0_n=1_ntokens=8192/data.json

printf "\n\n./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.4_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.4_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.4_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.4_n=64_ntokens=8192/data.json

#printf "\n\n./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.6_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/qwen-2.5-14b-instruct/aime_2024/temp=0.6_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.6_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-14b/QWEN14BRL/aime_2024/temp=0.6_n=64_ntokens=8192/data.json
