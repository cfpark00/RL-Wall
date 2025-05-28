import transformers
import datasets
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import glob

import sys
sys.path.append("/n/home12/cfpark00/ML/tools")

import lm_tools


if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="./data/sft/v5/toy-multistep-v5-16")
    parser.add_argument("--dataset_name", type=str, default="cfpark00/toy-multistep-v5-16")
    parser.add_argument("--i_ckpts", type=list, default=[-1])#[19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])#29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
    parser.add_argument("--eval_splits", type=list, default=["train_rl","test"])#["train","train_rl","test"])
    parser.add_argument("--temperatures", type=str, default="[0.025,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]")#"[0.025,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2]")
    parser.add_argument("--n_problems", type=int, default=512)
    parser.add_argument("--n_eval", type=int, default=128)
    args = parser.parse_args()

    folder=args.folder
    dataset_name=args.dataset_name
    i_ckpts=args.i_ckpts
    eval_splits=args.eval_splits
    temperatures=args.temperatures
    temperatures=eval(temperatures)
    n_problems=args.n_problems
    n_eval=args.n_eval
    
    model_names=glob.glob(f"{folder}/checkpoint-*")
    model_names.sort(key=lambda x: int(x.split("-")[-1]))
    model_names=[model_names[i] for i in i_ckpts]
    for model_name in model_names:
        print(f"Evaluating {model_name}")

        model=transformers.AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_name)
        model=model.to(device=device)
        dataset=datasets.load_dataset(dataset_name)

        eval_data={}
        for split in eval_splits:
            print(f"Split: {split}")
            ds=dataset[split].select(range(min(n_problems,len(dataset[split]))))
            prompts=ds["prompt"]
            prompt_token_ids,prompt_attention_masks=lm_tools.tokenize(tokenizer, texts=prompts,chunk_size=1024,padding_side="left",get_attention_mask=True,verbose=False)
            completions=ds["completion"]
            completion_token_ids,completion_attention_masks=lm_tools.tokenize(tokenizer, texts=completions,chunk_size=1024,padding_side="right",get_attention_mask=True,verbose=False)

            ds_torch=torch.utils.data.TensorDataset(prompt_token_ids,
                                                    prompt_attention_masks,
                                                    completion_token_ids,
                                                    completion_attention_masks)
            dataloader=torch.utils.data.DataLoader(ds_torch, batch_size=512, shuffle=False,drop_last=False)

            #t=0
            corrects_t0=[]
            for d in dataloader:
                token_ids=d[0].to("cuda")
                attention_masks=d[1].to("cuda")
                answer_ids=d[2].to("cuda")
                pred=model.generate(
                    input_ids=token_ids,
                    attention_mask=attention_masks,
                    do_sample=False,
                    max_new_tokens=answer_ids.shape[1],
                )[:,len(token_ids[0]):]
                
                error=(pred!=answer_ids)
                corrects_t0.extend(torch.all(~error,dim=1).cpu().numpy())
            corrects_t0=np.array(corrects_t0)
            print(f"pass@1 (t=0): {corrects_t0.mean():.4f}")
            
            corrects_temp={}
            for temperature in temperatures:
                dataloader=torch.utils.data.DataLoader(ds_torch, batch_size=128, shuffle=False,drop_last=False)
                corrects=[]
                for d in tqdm.tqdm(dataloader):
                    token_ids=d[0].to("cuda")
                    attention_masks=d[1].to("cuda")
                    answer_ids=d[2].to("cuda").repeat_interleave(n_eval,dim=0)
                    pred=model.generate(
                        input_ids=token_ids,
                        attention_mask=attention_masks,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=answer_ids.shape[1],
                        num_return_sequences=n_eval,
                    )[:,len(token_ids[0]):]
                    error=(pred!=answer_ids)
                    corrects.append(torch.all(~error,dim=1).reshape(-1,n_eval).cpu().numpy())
                corrects=np.concatenate(corrects,axis=0)
                corrects_temp[str(temperature)]=corrects
                print(f"pass@1 (t={temperature}): {corrects.mean():.4f}")
                coverage=np.any(corrects,axis=1).astype(np.float32).mean()
                print(f"pass@{n_eval} (t={temperature}): {coverage:.4f}")
            
            eval_data_element={
                "prompt": prompts,
                "completion": completions,
                "corrects_t0": corrects_t0,
                "corrects_temp": corrects_temp
            }

            eval_data[split]=eval_data_element

            eval_data_path=os.path.join(model_name, "eval_data_t.pt")
            torch.save(eval_data, eval_data_path)
            print(f" ")
