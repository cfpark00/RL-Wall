#math_train
##python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_train/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_train/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_train t=0.6
##python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_train/temp=0.6_n=16_ntokens=8192" --temperature 0.6 --n 16 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name math_train --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_train/temp=0.6_n=16_ntokens=8192" --temperature 0.6 --n 16 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

###
#math_500
##python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=0.1
python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.1_n=64_ntokens=8192" --temperature 0.1 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.1_n=64_ntokens=8192" --temperature 0.1 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=0.4
#python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.4_n=64_ntokens=8192" --temperature 0.4 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=0.6
##python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#math_500 t=1.0
##python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=1.0_n=64_ntokens=8192" --temperature 1.0 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=1.0_n=64_ntokens=8192" --temperature 1.0 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

###
#aime_2024
##python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/aime_2024/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/aime_2024/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"

#aime_2024 t=0.6
##python3 generate_responses.py --model_name qwen-2.5-7b-instruct --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/aime_2024/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
##python3 generate_responses.py --model_name qwen-7b-grpo-simplerl-40steps --dataset_name aime_2024 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/aime_2024/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"


#rl 100 steps
#math_500 t=0.0
python3 generate_responses.py --model_name qwen-7b-grpo-math-100steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.0_n=1_ntokens=8192" --temperature 0.0 --n 1 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#math_500 t=0.1
python3 generate_responses.py --model_name qwen-7b-grpo-math-100steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.1_n=64_ntokens=8192" --temperature 0.1 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#math_500 t=0.6
python3 generate_responses.py --model_name qwen-7b-grpo-math-100steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.6_n=64_ntokens=8192" --temperature 0.6 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"
#math_500 t=1.0
python3 generate_responses.py --model_name qwen-7b-grpo-math-100steps --dataset_name math_500 --exp_dir "./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=1.0_n=64_ntokens=8192" --temperature 1.0 --n 64 --max_tokens 8192 --tensor_parallel_size 2 --prompt_suffix "_boxed"




##correct
#math_train
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_train/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_train/temp=0.0_n=1_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_train/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_train/temp=0.0_n=1_ntokens=8192/data.json

#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_train/temp=0.6_n=16_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_train/temp=0.6_n=16_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_train/temp=0.6_n=16_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_train/temp=0.6_n=16_ntokens=8192/data.json

#math_500
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.0_n=1_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.0_n=1_ntokens=8192/data.json

printf "\n\n/data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.1_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.1_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.1_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.1_n=64_ntokens=8192/data.json

#printf "\n\n/data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.4_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.4_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.4_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.4_n=64_ntokens=8192/data.json

#printf "\n\n/data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.6_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=0.6_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.6_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=0.6_n=64_ntokens=8192/data.json

#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=1.0_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/math_500/temp=1.0_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=1.0_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/math_500/temp=1.0_n=64_ntokens=8192/data.json

#aime_2024
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/aime_2024/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/aime_2024/temp=0.0_n=1_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/aime_2024/temp=0.0_n=1_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/aime_2024/temp=0.0_n=1_ntokens=8192/data.json

#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/aime_2024/temp=0.6_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-2.5-7b-instruct/aime_2024/temp=0.6_n=64_ntokens=8192/data.json
#printf "\n\n./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/aime_2024/temp=0.6_n=64_ntokens=8192/data.json\n"
#python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-simplerl-40steps/aime_2024/temp=0.6_n=64_ntokens=8192/data.json

#rl 100 steps
printf "\n\n/data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.0_n=1_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.0_n=1_ntokens=8192/data.json
printf "\n\n/data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.1_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.1_n=64_ntokens=8192/data.json
printf "\n\n/data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.6_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=0.6_n=64_ntokens=8192/data.json
printf "\n\n/data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=1.0_n=64_ntokens=8192/data.json\n"
python3 extract_correct.py ./data/new_evals_03_03/qwen-7b/qwen-7b-grpo-math-100steps/math_500/temp=1.0_n=64_ntokens=8192/data.json