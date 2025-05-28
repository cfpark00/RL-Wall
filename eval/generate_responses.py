# Use a pipeline as a high-level helper
from transformers import AutoTokenizer
import vllm
import torch
from datasets import load_dataset
import json
import yaml
import os
import argparse
import tqdm
from vllm.lora.request import LoRARequest

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen-2.5-1.5b-instruct")
parser.add_argument("--dataset_name", type=str, default="math_parsable")
parser.add_argument(
    "--exp_dir",
    type=str,
    default="./data/baselines/qwen-2.5-1.5b-instruct/math_parsable/temp=1.0_seed=0",
)
parser.add_argument("--n", type=int, default=32)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max_tokens", type=int, default=16384)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--prompt_suffix", type=str, default="")
parser.add_argument('--tensor_parallel_size', type=int, default=2)
parser.add_argument("--inds", type=str, default=None)
parser.add_argument("--min_p", type=float, default=0.0)
parser.add_argument("--lora_path", type=str, default=None)
args = parser.parse_args()
# assemble into a yaml

model_path=utils.get_model_path(args.model_name)
#
ds=utils.get_dataset(args.dataset_name)
#
exp_dir = args.exp_dir
#
assert not os.path.exists(exp_dir), f"exp_dir {exp_dir} already exists"
os.makedirs(exp_dir, exist_ok=True)
#
n = args.n
#
temperature = args.temperature
#
max_tokens = args.max_tokens
#
seed = args.seed
assert seed is None, "vllm currently does not support different seed per sample"
#
system_prompt,prompt_format=utils.get_prompt_format(args.prompt_suffix)
#
tensor_parallel_size = args.tensor_parallel_size
#
inds = args.inds
if inds is not None:
    inds=[int(ind) for ind in inds.split(",")]
    ds = ds.select(inds)
#
min_p = args.min_p
#
lora_path = args.lora_path
if lora_path is not None:
    lora_request = LoRARequest("lora", 1, lora_path=lora_path)
else:
    lora_request = None

#####

# save all args as config.yaml in exp_dir
yaml_path = os.path.join(exp_dir, "config.yaml")
yaml.dump(vars(args), open(yaml_path, "w"))
save_path = os.path.join(exp_dir, "data.json")

# setup model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = vllm.LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16",gpu_memory_utilization=0.75,enable_lora=lora_request is not None)
# setting seed in sampling
sampling_params = vllm.SamplingParams(
    n=n, temperature=temperature, max_tokens=max_tokens, seed=seed, min_p=min_p
)

messagess = []
prompts=[]
for data in ds:
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
        ]
    else:
        messages = []
    prompt = prompt_format.replace("PROBLEM", data["problem"])
    messages.append({"role": "user", "content": prompt})
    textform = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    messagess.append(messages)
    prompts.append(textform)
#vllm call
outputss=model.generate(prompts, sampling_params, use_tqdm=True,lora_request=lora_request)

all_data = []
for data, messages, prompt, outputs in zip(ds, messagess, prompts,outputss):
    data["messages"] = messages
    data["prompt"] = prompt
    data["responses"] = [output.text for output in outputs.outputs]
    all_data.append(data)
json.dump(all_data, open(save_path, "w"), indent=2)
utils.free_vllm(model)
print("Done!")
