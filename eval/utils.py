from pydantic import BaseModel
import json
import numpy as np
import tqdm
from datasets import load_dataset
import sympy
from sympy.parsing.latex import parse_latex
import multiprocessing
import torch
import re
from typing import Optional


import verl_math_verifier

def get_model_path(model_name):
    model_paths = {
        "qwen-2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen-2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen-2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
        "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen-2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        "qwen-2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",

        "qwen-2.5-math-0.5b-instruct": "Qwen/Qwen2.5-Math-0.5B-Instruct",
        "qwen-2.5-math-1.5b-instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "qwen-2.5-math-7b": "Qwen/Qwen2.5-Math-7B",
        "qwen-2.5-math-7b-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",

        "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        
        "deepseek-math-7b-instruct": "deepseek-ai/deepseek-math-7b-instruct",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "allenai-olmo-2-1124-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct",

        "s1-32b":"simplescaling/s1-32B",

        "qwenmath-simplerl":"qwenmath-simplerl",

        ###custom trained models

        #0.5B RL
        "qwen-0.5b-grpo-math-10steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_10",
        "qwen-0.5b-grpo-math-20steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_20",
        "qwen-0.5b-grpo-math-30steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_30",
        "qwen-0.5b-grpo-math-40steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_40",
        "qwen-0.5b-grpo-math-50steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_50",
        "qwen-0.5b-grpo-math-60steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_60",
        "qwen-0.5b-grpo-math-70steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_70",
        "qwen-0.5b-grpo-math-80steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_80",
        "qwen-0.5b-grpo-math-90steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_90",
        "qwen-0.5b-grpo-math-100steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-0.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_100",


        #Math-1.5B RL
        "qwen-2.5-math-1.5b-instruct-grpo-v1-5": "/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-Math-1.5B-Instruct_openrlhf_grpo_MATH_train/episode5_hf",
        "qwen-2.5-math-1.5b-instruct-grpo-v1-10": "/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-Math-1.5B-Instruct_openrlhf_grpo_MATH_train/episode10_hf",
        "qwen-2.5-math-1.5b-instruct-grpo-v1-18": "/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-Math-1.5B-Instruct_openrlhf_grpo_MATH_train/episode18_hf",

        #1.5B RL
        "qwen-2.5-1.5b-instruct-grpo-gsm8k-v1-7":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_openrlhf_grpo_GSM8K_train/episode7_hf",
        "qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-7":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_openrlhf_ppo_GSM8K_train/episode7_hf",
        "qwen-2.5-1.5b-instruct-grpo-math-v1-130":"./data/models/qwen-2.5-1.5b-instruct-grpo-math-v1-130/state_dict_12-hf",
        "qwen-2.5-1.5b-instruct-ppo-math-v1-130":"./data/models/qwen-2.5-1.5b-instruct-ppo-math-v1-130/state_dict_12-hf",

        #1.5B RL dynamics
        "qwen-1.5b-grpo-math-10steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_10",
        "qwen-1.5b-grpo-math-20steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_20",
        "qwen-1.5b-grpo-math-30steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_30",
        "qwen-1.5b-grpo-math-40steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_40",
        "qwen-1.5b-grpo-math-50steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_50",
        "qwen-1.5b-grpo-math-60steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_60",
        "qwen-1.5b-grpo-math-70steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_70",
        "qwen-1.5b-grpo-math-80steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_80",
        "qwen-1.5b-grpo-math-90steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_90",
        "qwen-1.5b-grpo-math-95steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_95",

        #subjects 1.5B
        "qwen-1.5b-grpo-math-algebra":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train_algebra/hf_model/global_step_99/",
        "qwen-1.5b-grpo-math-counting":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train_counting/hf_model/global_step_99/",
        "qwen-1.5b-grpo-math-geometry":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train_geometry/hf_model/global_step_99/",
        "qwen-1.5b-grpo-math-interalgebra":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train_interalgebra/hf_model/global_step_96/",
        "qwen-1.5b-grpo-math-number":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train_number/hf_model/global_step_99/",
        "qwen-1.5b-grpo-math-prealgebra":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train_prealgebra/hf_model/global_step_99/",
        "qwen-1.5b-grpo-math-precalculus":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train_precalculus/hf_model/global_step_99/",

        "qwen-1.5b-sft-math-v1-50":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_sft_MATH_train/checkpoint-50",
        "qwen-1.5b-sft-math-v1-100":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_sft_MATH_train/checkpoint-100",
        "qwen-1.5b-sft-math-v1-250":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_sft_MATH_train/checkpoint-250",
        "qwen-1.5b-sft-math-v1-400":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_sft_MATH_train/checkpoint-400",

        "qwen-1.5b-grpo-sft-math-v1-100":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-GRPO-sft-MATH/checkpoint-100",
        "qwen-1.5b-grpo-sft-math-v1-200":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-GRPO-sft-MATH/checkpoint-200",
        "qwen-1.5b-grpo-sft-math-v1-300":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-GRPO-sft-MATH/checkpoint-300",
        "qwen-1.5b-grpo-sft-math-v1-400":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-GRPO-sft-MATH/checkpoint-400",
        "qwen-1.5b-grpo-sft-math-v1-500":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-GRPO-sft-MATH/checkpoint-500",
        "qwen-1.5b-grpo-sft-math-v1-600":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-GRPO-sft-MATH/checkpoint-600",
        "qwen-1.5b-grpo-sft-math-v1-700":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-GRPO-sft-MATH/checkpoint-700",

        "qwen-2.5-1.5b-instruct-lin":"/n/netscratch/dam_lab/Everyone/wall/cfpark00/sft_models/math_linearized_random_8_1",
        "qwen-2.5-1.5b-instruct-lin-v2":"/n/netscratch/dam_lab/Everyone/wall/cfpark00/sft_models/math_linearized_correct_last_8_1",
        "qwen-2.5-1.5b-instruct-lin-v3":"/n/netscratch/dam_lab/Everyone/wall/cfpark00/sft_models/gsm8k_linearized_random_4_1",
        "qwen-2.5-1.5b-instruct-lin-v4":"/n/netscratch/dam_lab/Everyone/wall/cfpark00/sft_models/gsm8k_linearized_correct_last_4_1",

        "qwen-2.5-7b-instruct-lin-v1":"/n/netscratch/dam_lab/Everyone/wall/cfpark00/sft_models/qwen_7b_gsm8k_linearized_random_4_1",
        "qwen-2.5-7b-instruct-lin-v2-1":"/n/home12/cfpark00/ML/llm-meta-rl/data/sft/gsm8k_linearized_7b/random_4_small/checkpoint-94",
        "qwen-2.5-7b-instruct-lin-v2-2":"/n/home12/cfpark00/ML/llm-meta-rl/data/sft/gsm8k_linearized_7b/random_4_small/checkpoint-141",
        "qwen-2.5-7b-instruct-lin-v2-3":"/n/home12/cfpark00/ML/llm-meta-rl/data/sft/gsm8k_linearized_7b/random_4_small/checkpoint-329",

        #1.5B s1 sft
        "qwen-2.5-1.5b-instruct-s1sft-v1-200":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-sft-s1/checkpoint-200",

        #3b
        "qwen-3b-grpo-math-20steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/hf_model/global_step_20",
        "qwen-3b-grpo-math-60steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/hf_model/global_step_60",
        "qwen-3b-grpo-math-100steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/hf_model/global_step_100",
        "qwen-3b-grpo-math-140steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/hf_model/global_step_140",
        "qwen-3b-grpo-math-180steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/hf_model/global_step_180",
        "qwen-3b-grpo-math-200steps":"/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/hf_model/global_step_200",

        #7b simplerl
        "qwen-7b-grpo-simplerl-40steps":"/n/netscratch/dam_lab/Everyone/wall/cfpark00/simplerl/Qwen2.5-7B-Instruct_verl_grpo_SIMPLERL_train/hf_model/global_step_40",
        
        "qwen-7b-grpo-math-100steps": "/n/netscratch/dam_lab/Everyone/wall/cfpark00/models/Qwen2.5-7B-Instruct_verl_grpo_MATH_train/global_step_100/actor/state_dict-hf",
    }
    return model_paths[model_name]

def get_dataset(dataset_name):
    if dataset_name == "aime_2024":
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        ds = ds.filter(lambda data: "2024_AIME" in data["url"])
    elif dataset_name == "math_500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.rename_column("unique_id", "id")
    elif dataset_name == "math_parsable":
        ds = load_dataset("cfpark00/MATH", split="test_parsable")
        ds = ds.rename_column("unique_id", "id")
    elif dataset_name == "math_train":
        ds = load_dataset("/n/home12/cfpark00/ML/llm-meta-rl/data/datasets/math", split="train")
        ds = ds.rename_column("unique_id","id")
    elif dataset_name == "math_test":
        ds = load_dataset("/n/home12/cfpark00/ML/llm-meta-rl/data/datasets/math", split="test")
        ds = ds.rename_column("unique_id","id")
    elif dataset_name == "gsm8k_train":
        ds=load_dataset("openai/gsm8k","main",split="train")#"test"
        ds=ds.rename_column("question","problem")
        ds=ds.rename_column("answer","solution")
        def get_answer(x):
            return {"answer":x["solution"].split("####")[-1].strip()}
        ds=ds.map(get_answer)
        return ds
    else:
        ds = load_dataset(dataset_name, split="test")
        ds = ds.rename_column("unique_id","id")
    return ds

def get_prompt_format(prompt_suffix):
    if prompt_suffix == "":
        system_prompt = None
        prompt_format = """PROBLEM"""
    elif prompt_suffix == "_boxed":
        system_prompt = None
        prompt_format = """Answer the following math problem. Please reason step by step, and put your final answer within \\boxed{}.\n\nPROBLEM"""
    elif prompt_suffix == "_boxed2":
        system_prompt = None
        prompt_format = """Please reason step by step, and put your final answer within \\boxed{}. PROBLEM"""
    elif prompt_suffix == "_sunnyformat":
        system_prompt ="Please reason step by step, and put your final answer within \\boxed{}."
        prompt_format = """PROBLEM"""
    else:
        raise ValueError(f"Unknown prompt_suffix: {prompt_suffix}")
    return system_prompt, prompt_format

def free_vllm(llm):
    import gc
    import torch
    from vllm.distributed.parallel_state import destroy_model_parallel

    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()

######

def verl_batch_verifier(responses,gt_answer,return_answers=False):
    model_answers=[]
    corrects = []
    for response in responses:
        if return_answers:
            answer=verl_math_verifier.get_answer(response)
            model_answers.append(answer)
        correct = verl_math_verifier.compute_score(response, gt_answer)>0.5#verl returns 1.0 for correct and 0.0 for incorrect
        corrects.append(correct)
    if return_answers:
        return corrects,model_answers
    return corrects

def gpt_verifier(
    client,
    model_name,
    model_answers,
    gt_answers,
    batch_size=10,
    verbose=True,
    n_max_trials=5,
):
    class Corrects(BaseModel):
        equivalents: list[bool]

    system_prompt = (
        'Given two list of answers, determine if they are mathematically equivalent per element.\n'
        'Be generous, for example:\n'
        ' - "$2$" and "2.0"\n'
        ' - "$x=3.5$" and "7/2"\n'
        ' - "$-1*3/2$" and "$-1.5$"\n'
        ' - "The answer is: 30 cm." and "30"\n'
        'are all equivalent.\n\n'
        'Output as a json with a list of booleans as a field "equivalents".'
    )
    n_batches = np.ceil(len(model_answers) / batch_size).astype(int)
    corrects = []
    for i_batch in tqdm.tqdm(range(n_batches), disable=not verbose):
        i_start = i_batch * batch_size
        i_end = min((i_batch + 1) * batch_size, len(model_answers))
        n_b = len(model_answers[i_start:i_end])
        mas_str = str(model_answers[i_start:i_end])
        ga_str = str(gt_answers[i_start:i_end])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "List 1:\n" + mas_str + "\nList 2:\n" + ga_str},
        ]
        done = False
        n_tried = 0
        while not done:
            #response = language_models.get_message(
            #    model_name_or_spec=model_name_or_spec,
            #    messages=messages,
            #    response_format=Corrects,
            #)
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=Corrects,
            )
            corrects_ = completion.choices[0].message.parsed.equivalents
            if len(corrects_) == n_b:
                done = True
            else: 
                n_tried += 1
                if n_tried >= n_max_trials:
                    print("Failed to get correct number of responses, trying one by one")
                    #do one by one
                    corrects_ = []
                    for i in range(i_start,i_end):
                        corrects_.append(gpt_single_verifier(
                            client=client,
                            model_name=model_name,
                            model_answer=model_answers[i],
                            gt_answer=gt_answers[i]
                        ))
                    assert len(corrects_) and all([c in [True,False] for c in corrects_])
                    done=True
        corrects.extend(corrects_)
    return corrects

def gpt_single_verifier(client, model_name, model_answer, gt_answer):
    class Correct(BaseModel):
        equivalent: bool
    system_prompt = (
        'Given two answers, determine if they are mathematically equivalent.\n'
        'Be generous, for example:\n'
        ' - "$2$" and "2.0"\n'
        ' - "$x=3.5$" and "7/2"\n'
        ' - "$-1*3/2$" and "$-1.5$"\n'
        ' - "The answer is: 30 cm." and "30"\n'
        ' - " is 2. So the final answer is 6" and "6 kilometers"\n'
        'are all equivalent.\n\n'
        'Output as a json with a field "equivalent(boolean)".'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Answer 1:\n" + model_answer + "\nAnswer 2:\n" + gt_answer},
    ]
    #single shot and if not just make False
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=Correct,
    )
    equivalent = completion.choices[0].message.parsed.equivalent
    if equivalent not in [True,False]:
        equivalent=False
    return equivalent

def gpt_batch_verifier(
    client,
    model_name,
    model_answers,
    gt_answer,
    minibatch_size=8,
    n_max_trials=5,
):
    class Corrects(BaseModel):
        corrects: list[bool]

    system_prompt = (
        'You are given a list of answers from many students and a ground truth (GT) answer. '
        'For each student answer, determine if it is mathematically equivalent to the GT answer. '
        'Be generous, for example:\n'
        ' - "$2$" and "2.0"\n'
        ' - "$x=3.5$" and "7/2"\n'
        ' - "$-1*3/2$" and "$-1.5$"\n'
        ' - "The answer is: 30 cm." and "30"\n'
        ' - "nge is 2. So the final answer is 6" and "6 kilometers"\n'
        'are all equivalent.\n\n'
        'Output as a json with a field: "corrects (list of booleans)".'
    )
    corrects=[]
    n_minibatches = np.ceil(len(model_answers) / minibatch_size).astype(int)
    for i_batch in range(n_minibatches):
        model_answers_=model_answers[i_batch * minibatch_size : (i_batch + 1) * minibatch_size]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Student answers:\n"
                + str(model_answers_)
                + "\nGT answer:\n"
                + str(gt_answer),
            },
        ]
        done = False
        n_tried = 0
        while not done:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=Corrects,
            )
            corrects_ = completion.choices[0].message.parsed.corrects
            if len(corrects_) == len(model_answers_):
                done = True
            else:
                n_tried += 1
                if n_tried >= n_max_trials:#as above do everything one by one
                    print("Failed to get correct number of responses, trying one by one")
                    corrects_ = []
                    for model_answer in model_answers_:
                        corrects_.append(gpt_single_verifier(
                            client=client,
                            model_name=model_name,
                            model_answer=model_answer,
                            gt_answer=gt_answer
                        ))
                    assert len(corrects_) and all([c in [True,False] for c in corrects_])
                    done=True
        corrects.extend(corrects_)
    return corrects













######

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] != left:
            return s
        return s[len(left) :]

    left = "\\boxed{"
    if (s[: len(left)] != left) or (s[-1] != "}"):
        return s

    return s[len(left) : -1]

def get_answer(response, max_answer_length=None):
    ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"
    a=response

    if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
    # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
    elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
        a = matches[-1]  # Get the last match
    if max_answer_length is not None and len(a) > max_answer_length:
        return a[-max_answer_length:]
    return a
    

def gpt_batch_get_wrong_majority(
    client,
    model_name,
    model_answers,
    gt_answer
):
    class WrongMajority(BaseModel):
        wrong_majority: str

    system_prompt = (
        'You are given a list of student answers to a math question and a ground truth (GT) answer. '
        'You have this rather peculiar task: find the *most common wrong answer* among the student answers. '
        'Be generous about what is "right":\n'
        ' - "$2$" and "2.0"\n'
        ' - "$x=3.5$" and "7/2"\n'
        ' - "$-1*3/2$" and "$-1.5$"\n'
        ' - "The answer is: 30 cm." and "30"\n'
        ' - "nge is 2. So the final answer is 6" and "6 kilometers"\n'
        'are all fine.\n'
        'Thus, if the list of answer is ["2.0", "2", "two", "two", "2.0", "2.0","1"], and the GT answer is "2", '
        'the most common wrong answer is "1".\n\n'
        'If all answers are correct, output an empty string, "".\n\n'
        'Output as a json with a field: "wrong_majority" (string).'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Student answers:\n"
            + str(model_answers)
            + "\nGT answer:\n"
            + str(gt_answer),
        },
    ]
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=WrongMajority,
    )
    wrong_majority = completion.choices[0].message.parsed.wrong_majority
    return wrong_majority



def get_linearized_response(responses,n_max_reps=16,corrects=None,correct_last=False,seed=None):
    if seed is not None:
        np.random.seed(seed)
    backtracking_prompts=[
        " Let's try again to make sure. ",
        "\nTo make sure, let's try again.\n",
        "\n\nLet's try again.\n\n",
        " Wait, let me try again. ",
        "\nWait, let's try again, just in case.\n",
        "\n\nWait, let's make sure by trying again.\n\n",
        " Let's verify by trying again. ",
        " Just to make sure, let's try again. ",
        "\nMaybe I made a mistake. Let's try again.\n",
        "\n\nPerhaps this is correct, but let's try again to make sure.\n\n",
        " Let me try a bit differently to confirm. ",
        "\nI'm not sure about this I'll try again.\n",
        "\nIf I didn't respect the format, I should correct that. Let's try again.\n",
        " I should make sure to get this right. ",
        "\n\nI should get the answer and the format right. Let's try again.\n\n",
    ]
    if correct_last:
        corrects=np.array(corrects)
        n_corrects=sum(corrects)
        if n_corrects==0:
            return np.random.choice(responses)+" But I think I made a mistake."
        i_correct_response=np.random.choice(np.where(corrects)[0])
        correct_response=responses[i_correct_response]
        n_incorrects=len(corrects)-n_corrects
        if n_incorrects==0:
            return correct_response
        n_reps=np.random.randint(0,min(n_incorrects,n_max_reps-1)+1)
        i_incorrect_responses=np.random.choice(np.where(~corrects)[0],n_reps,replace=False)
        response=""
        for i_incorrect_response in i_incorrect_responses:
            response+=responses[i_incorrect_response]
            response+=np.random.choice(backtracking_prompts)
        response+=correct_response
        return response
    else:
        n_reps=np.random.randint(1,min(len(responses),n_max_reps)+1)
        i_responses=np.random.choice(len(responses),n_reps,replace=False)
        response=""
        for ii,i_response in enumerate(i_responses):
            response+=responses[i_response]
            if ii!=(n_reps-1):
                response+=np.random.choice(backtracking_prompts)
        return response



###########
######### old code




def get_answer_old(s, max_answer_length=None):
    start = s.rfind(r"\boxed{")
    if start == -1:
        return "ERROR"
    start += len(r"\boxed{")
    level = 1
    i = start
    out = []
    while i < len(s) and level > 0:
        if s[i] == "{":
            level += 1
        elif s[i] == "}":
            level -= 1
        if level > 0:
            out.append(s[i])
        i += 1
    ans = "".join(out)
    if level != 0:
        return "ERROR"
    if max_answer_length is not None and len(ans) > max_answer_length:
        return "ERROR"
    return ans

###
ERROR_STR = "ERROR"


def simplify(expr):
    if expr == ERROR_STR:
        return None
    try:
        expr = parse_latex(expr)
        simplified = sympy.simplify(expr)
        return simplified
    except Exception as e:
        return None


def batch_parallel_simplify(answers, timeout_per_task=1, verbose=True):
    simplified = [None] * len(answers)
    num_workers = multiprocessing.cpu_count()
    pbar = tqdm.tqdm(total=len(answers), disable=not verbose)
    func = simplify
    with multiprocessing.Pool(processes=num_workers) as pool:
        async_results = []
        for idx, ans in enumerate(answers):
            async_result = pool.apply_async(func, args=(ans,))
            async_results.append((idx, async_result))

        for idx, async_result in async_results:
            # force set pbar to idx
            pbar.n = idx
            pbar.refresh()
            try:
                result = async_result.get(timeout=timeout_per_task)
                simplified[idx] = result
            except multiprocessing.TimeoutError:
                # If the task times out, mark as False
                simplified[idx] = None
            except Exception as e:
                # For any other exceptions, also mark as False
                simplified[idx] = None
    pbar.close()
    return simplified

def same(expr1, expr2):
    try:
        return sympy.simplify(expr1 - expr2) == 0
    except Exception as e:
        return False

def compare_pair(model_ans, gt_ans):
    try:
        expr1 = parse_latex(model_ans)
        simplified_expr1 = sympy.simplify(expr1)
    except Exception as e:
        return False
    try:
        expr2 = parse_latex(gt_ans)
        simplified_expr2 = sympy.simplify(expr2)
    except Exception as e:
        return False

    try:
        return sympy.simplify(simplified_expr1 - simplified_expr2) == 0
    except Exception as e:
        return False


def sympy_parallel_verifier(model_answers, gt_answers, timeout_per_task=1,verbose=True):
    corrects = [False] * len(model_answers)
    num_workers = multiprocessing.cpu_count()
    pbar = tqdm.tqdm(total=len(model_answers), disable=not verbose)
    with multiprocessing.Pool(processes=num_workers) as pool:
        async_results = []
        for idx, (model_ans, gt_ans) in enumerate(zip(model_answers, gt_answers)):
            async_result = pool.apply_async(compare_pair, args=(model_ans, gt_ans))
            async_results.append((idx, async_result))

        for idx, async_result in async_results:
            # force set pbar to idx
            pbar.n = idx
            pbar.refresh()
            try:
                result = async_result.get(timeout=timeout_per_task)
                corrects[idx] = result
            except multiprocessing.TimeoutError:
                # If the task times out, mark as False
                corrects[idx] = False
            except Exception as e:
                # For any other exceptions, also mark as False
                corrects[idx] = False
    pbar.close()
    return corrects

def sympy_parallel_voter(model_answers):#reduces everything to a canonical form and then votes among the valid ones
    simplified=batch_parallel_simplify(model_answers, timeout_per_task=2, verbose=False)
    votes={}
    for answer,simplified in zip(model_answers,simplified):
        voted=False
        for key,value in votes.items():
            if same(simplified,value[1]):
                votes[key][0]+=1
                voted=True
                break
        if not voted:
            votes[answer]=[1,simplified]
    for key in votes.keys():
        votes[key]=votes[key][0]
    return votes

###

def reward_func(completions, **kwargs):
    gt_answers = kwargs["answer"]
    max_answer_length = kwargs.get("max_answer_length", 100)
    answers = [
        get_answer(completion[-1]["content"], max_answer_length=max_answer_length)
        for completion in completions
    ]
    answers.extend(gt_answers)  # gt answers is a list now
    answers_simplified = batch_parallel_simplify(
        answers, timeout_per_task=6, verbose=False
    )
    model_answers_simplified = answers_simplified[: len(completions)]
    gt_answers_simplified = answers_simplified[len(completions) :]
    # print("Simplified Model Answers:", model_answers_simplified)
    # print("Simplified GT Answers:", gt_answers_simplified)

    #removing the assert since it might halt a long run
    #assert all(
    #    [
    #        gt_answer_simplified is not None
    #        for gt_answer_simplified in gt_answers_simplified
    #    ]
    #), gt_answers
    rewards = []
    for model_answer, gt_answer in zip(model_answers_simplified, gt_answers_simplified):
        if model_answer is None:
            rewards.append(0)
        elif gt_answer is None:#i guess we need to do this for now.
            rewards.append(0)
        elif same(model_answer, gt_answer):
            rewards.append(1)
        else:
            rewards.append(0)
    return rewards


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import signal

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def batch_simplify(answers, timeout_per_task=1, verbose=True):#not parallel but with timeout using signal
    simplified = [None] * len(answers)
    pbar = tqdm.tqdm(total=len(answers), disable=not verbose)
    for idx, ans in enumerate(answers):
        try:
            with timeout(seconds=timeout_per_task):
                result = simplify(ans)
        except Exception as e:
            result = None
        simplified[idx] = result
        pbar.n = idx
        pbar.refresh()
    pbar.close()
    return simplified

def sympy_verifier(model_answers, gt_answers, timeout_per_task=1, verbose=True):#not parallel but with timeout using signal
    corrects = [False] * len(model_answers)
    pbar = tqdm.tqdm(total=len(model_answers), disable=not verbose)
    for idx, (model_ans, gt_ans) in enumerate(zip(model_answers, gt_answers)):
        try:
            with timeout(seconds=timeout_per_task):
                result = compare_pair(model_ans, gt_ans)
        except Exception as e:
            result = False
        corrects[idx] = result
        pbar.n = idx
        pbar.refresh()
    pbar.close()
    return corrects

def sympy_voter(model_answers, timeout_per_task=1, verbose=True):#not parallel but with timeout using signal
    simplifieds = batch_simplify(model_answers, timeout_per_task=timeout_per_task, verbose=verbose)
    votes={None:[[],None]}
    for i,(answer,simplified) in enumerate(zip(model_answers,simplifieds)):
        if simplified is None:
            votes[None][0].append(i)
            continue
        voted=False
        for key,value in votes.items():
            if same(simplified,value[1]):
                votes[key][0].append(i)
                voted=True
                break
        if not voted:
            votes[answer]=[[i],simplified]
    for key in votes.keys():
        votes[key]=votes[key][0]
    return votes

