import argparse
import os
import json
import wandb  # if you want to keep logging with wandb
import torch
import datasets
import yaml
#import tqdm
from accelerate.utils import tqdm
import shutil
import numpy as np
import glob
import os
#for peft
from peft import LoraConfig, get_peft_model, TaskType

from accelerate import Accelerator
accelerator = Accelerator()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
)

import sys
sys.path.append("/n/home12/cfpark00/ML/tools")
import lm_tools

def check_config(config):
    assert "model_name" in config, "Model name not found in config"
    assert "dataset_path" in config, "Dataset path not found in config"
    assert "save_path" in config, "Model save path not found in config"

def get_per_device_batch_size(config):
    num_gpus = accelerator.num_processes
    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["training_arguments"]["gradient_accumulation_steps"]
    # Set up the effective batch size
    if not config["model_parallel"]:
        assert batch_size % num_gpus == 0, "Batch size should be divisible by the number of GPUs"
        per_device_batch_size = batch_size // num_gpus
        accelerator.print("Effective batch size:", batch_size)
        accelerator.print("Per device batch size:", per_device_batch_size)
        accelerator.print("N samples per gradient:", batch_size * gradient_accumulation_steps)
    else:
        per_device_batch_size = batch_size
        accelerator.print("Using",num_gpus,"GPUs for model parallelism")
        accelerator.print("Effective batch size:", batch_size)
        accelerator.print("Per device batch size:", per_device_batch_size)
        accelerator.print("N samples per gradient:", batch_size * gradient_accumulation_steps)
    return per_device_batch_size

def get_train_dataset(config,tokenizer,chunk_size=1000):
    dataset_path = config["dataset_path"]
    dataset_split = config.get("dataset_split", None)
    use_assistant_mask = config["use_assistant_mask"]
    instruct_mode = config["instruct_mode"]
    # Load the dataset
    accelerator.print("Loading dataset...")
    if dataset_path.endswith(".json"):
        assert dataset_split is None, "Dataset split should not be provided for a JSON dataset"
        assert os.path.exists(dataset_path), "Dataset path does not exist"
        messagess = json.load(open(dataset_path, "r"))
    else:
        assert dataset_split is not None, "Dataset split should be provided for a Hugging Face dataset"
        dataset = datasets.load_dataset(dataset_path, split=dataset_split)
        if instruct_mode:
            messagess = [dataset[i]["messages"] for i in range(len(dataset))]
        else:
            messagess = [dataset[i]["text"] for i in range(len(dataset))]
    accelerator.print("Number of messages:", len(messagess))
    if instruct_mode:
        assert isinstance(messagess, list) and isinstance(messagess[0], list) and isinstance(messagess[0][0], dict), "Dataset should be a list of messages (list of dicts)"
    else:
        assert isinstance(messagess, list) and isinstance(messagess[0], str), "Dataset should be a list of strings"
    accelerator.print("Example message:", messagess[0])

    # Process and tokenize the messages
    train_max_length = config.get("train_max_length",None)

    if use_assistant_mask:
        assert instruct_mode, "Assistant mask can only be used in instruct mode"
        token_idss = []
        assistant_masks = []
        for messages in tqdm(messagess, desc="Tokenizing messages..."):
            token_ids, assistant_mask = lm_tools.tokenize_with_assistant_mask(tokenizer, messages)
            token_idss.append(token_ids)
            assistant_masks.append(assistant_mask)
        token_idss,assistant_masks=lm_tools.pad_sequences(token_idss, tokenizer, assistant_masks=assistant_masks)
        if train_max_length is not None:
            token_idss=token_idss[:,:train_max_length]
            assistant_masks=assistant_masks[:,:train_max_length]
        accelerator.print("Token idss shape:", token_idss.shape)
        accelerator.print("Assistant masks shape:", assistant_masks.shape)
    else:
        if instruct_mode:
            token_idss = lm_tools.apply_chat_template(tokenizer, messagess, chunk_size=chunk_size)
        else:
            token_idss = lm_tools.tokenize(tokenizer=tokenizer, texts=messagess, chunk_size=chunk_size)
        if train_max_length is not None:
            token_idss=token_idss[:,:train_max_length]
        accelerator.print("Token idss shape:", token_idss.shape)


    labels = token_idss.clone()
    
    if use_assistant_mask:
        labels[~assistant_masks] = -100

    # Build the training dataset (a list of dictionaries)
    train_dataset = [{"input_ids": token_idss[i], "labels": labels[i]} for i in range(token_idss.size(0))]
    return train_dataset

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_config(yaml_path=None,overwrite=False):
    if yaml_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("yaml_path", type=str)
        parser.add_argument("--overwrite", action="store_true")
        args = parser.parse_args()
        yaml_path = args.yaml_path
        overwrite = args.overwrite

    assert os.path.exists(yaml_path), "YAML file does not exist"
    input_config = yaml.safe_load(open(yaml_path, "r"))
    config = {
        "batch_size": 8,
        "model_parallel": False,
        "use_assistant_mask": False,
        "instruct_mode": True,
        "training_arguments": {
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 1,
            "lr_scheduler_type": "constant",
            "learning_rate": 0.0003,
            "logging_steps": 10,
            "bf16": True,
            "save_only_model": True,
            "report_to": "wandb",
        },
        "overwrite":overwrite
    }
    config = recursive_dict_update(config, input_config)
    check_config(config)
    return config

def get_model_tokenizer(config):
    model_name = config["model_name"]
    # Load model and tokenizer
    accelerator.print("Loading model and tokenizer...")
    dtype=torch.bfloat16 if config["training_arguments"]["bf16"] else torch.float32
    if config["model_parallel"]:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="right")

    if "lora_params" in config:
        accelerator.print("Using PEFT model...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **config["lora_params"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

def main(config):
    if config["overwrite"] and os.path.exists(config["save_path"]):
        print("Removing:",config["save_path"])
        shutil.rmtree(config["save_path"])
    if "wandb" in config:
        use_wandb = True
        wandb_project_name = config["wandb"]["project_name"]
        os.environ["WANDB_PROJECT"] = wandb_project_name
        wandb_run_name = config["wandb"]["run_name"]
        config["training_arguments"]["report_to"] = "wandb"
        config["training_arguments"]["run_name"] = wandb_run_name

    save_path = config["save_path"]

    per_device_batch_size = get_per_device_batch_size(config)

    model,tokenizer=get_model_tokenizer(config)

    train_dataset = get_train_dataset(config,tokenizer)

    # Set up TrainingArguments.
    # Note: Hugging Face Trainer uses Accelerate under the hood,
    # so no explicit Accelerator initialization is required.
    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=per_device_batch_size,
        **config["training_arguments"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()

    # Save the final model and tokenizer
    final_model_path = os.path.join(save_path, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    #save tokenizer to all ckpts
    ckpt_paths=glob.glob(os.path.join(save_path,"checkpoint-*"))
    for ckpt_path in ckpt_paths:
        tokenizer.save_pretrained(ckpt_path)
    accelerator.print("Models saved at:", save_path)
    accelerator.print("Done!")

if __name__ == "__main__":
    config = get_config()
    main(config)