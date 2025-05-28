#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-0.5b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-0.5b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method verl_batched
python3 calc_response_lengths.py ./data/new_evals_03_03/qwen-2.5-0.5b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json "Qwen/Qwen2.5-0.5B-Instruct"

#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method verl_batched
python3 calc_response_lengths.py ./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json "Qwen/Qwen2.5-1.5B-Instruct"

#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-3b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-3b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method verl_batched
python3 calc_response_lengths.py ./data/new_evals_03_03/qwen-2.5-3b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json "Qwen/Qwen2.5-3B-Instruct"

#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-7b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-7b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method verl_batched
python3 calc_response_lengths.py ./data/new_evals_03_03/qwen-2.5-7b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json "Qwen/Qwen2.5-7B-Instruct"

#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-14b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-14b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method verl_batched
python3 calc_response_lengths.py ./data/new_evals_03_03/qwen-2.5-14b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json "Qwen/Qwen2.5-14B-Instruct"

#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-32b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/qwen-2.5-32b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method verl_batched
python3 calc_response_lengths.py ./data/new_evals_03_03/qwen-2.5-32b-instruct/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json "Qwen/Qwen2.5-32B-Instruct"

#python3 extract_correct.py ./data/new_evals_03_03/s1-32b/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method gpt_batched
#python3 extract_correct.py ./data/new_evals_03_03/s1-32b/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json --method verl_batched
python3 calc_response_lengths.py ./data/new_evals_03_03/s1-32b/math_500/temp\=1.0_n\=8_ntokens\=8192/data.json "simplescaling/s1-32B"



