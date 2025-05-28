import json
import argparse
import tqdm
import os
import shutil
import numpy as np

import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    parser.add_argument(
        "--method", type=str, default="verl_batched", choices=["gpt","gpt_batched","verl_batched"]
    )
    parser.add_argument(
        "--response_key", type=str, default="responses"
    )
    parser.add_argument("--minibatch_size", type=int, default=8)
    args = parser.parse_args()
    result_path = args.result_path
    minibatch_size = args.minibatch_size
    method = args.method
    response_key = args.response_key
    save_key="corrects_"+method+"_"+response_key

    assert os.path.exists(result_path), "Result file not found: {}".format(result_path)
    backup_dir = os.path.join(os.path.dirname(result_path), "backup")
    os.makedirs(backup_dir, exist_ok=True)
    # copy the original file to backup
    shutil.copy(result_path, os.path.join(backup_dir, os.path.basename(result_path)))

    ####
    data = json.load(open(result_path, "r"))
    all_corrects = []
    if method == "gpt":
        from openai import OpenAI
        client = OpenAI()

        model_answers_all=[]
        gt_answers_all=[]
        for element in tqdm.tqdm(data,desc="Collecting model answers"):
            responses = element[response_key]
            model_answers = [
                utils.get_answer(response, max_answer_length=50) for response in responses
            ]
            gt_answer = element["answer"]
            model_answers_all.extend(model_answers)
            gt_answers_all.extend([gt_answer]*len(model_answers))
        print("Processing...")
        corrects=utils.gpt_verifier(
                client=client,
                model_name="gpt-4o-mini",
                model_answers=model_answers_all,
                gt_answers=gt_answers_all,
                batch_size=minibatch_size,
                verbose=True,
                n_max_trials=5,
            )
        all_corrects.extend(corrects.reshape(len(data),-1))
        assert len(corrects)==len(model_answers_all)
        head=0
        for element in data:
            n=len(element[response_key])
            element[save_key]=corrects[head:head+n]
            head+=n
        assert head==len(corrects)

    elif method == "gpt_batched":
        from openai import OpenAI
        client = OpenAI()

        for element in tqdm.tqdm(data,desc="Processing"):
            responses = element[response_key]
            model_answers = [
                utils.get_answer(response, max_answer_length=50) for response in responses
            ]
            gt_answer = element["answer"]
            corrects=utils.gpt_batch_verifier(
                client=client,
                model_name="gpt-4o-mini",
                model_answers=model_answers,
                minibatch_size=minibatch_size,
                gt_answer=gt_answer,
                n_max_trials=5,
            )
            element[save_key]=corrects
            all_corrects.append(corrects)

    elif method == "verl_batched":
        #verl_batch_verifier(responses,gt_answer):
        for element in tqdm.tqdm(data,desc="Processing"):
            responses = element[response_key]
            gt_answer = element["answer"]
            corrects,model_answers=utils.verl_batch_verifier(responses,gt_answer,return_answers=True)
            element[save_key]=corrects
            element["verl_model_answers"]=model_answers
            all_corrects.append(corrects)
    else:
        raise ValueError("Unknown correct method: {}".format(method))
    
    all_corrects = np.array(all_corrects).astype(np.float32)
    print("Overall coverage: {:.3f}".format(np.mean((all_corrects.max(1)>0.0).astype(np.float32))))
    print("Overall accuracy: {:.3f}".format(np.mean(all_corrects)))

    json.dump(data, open(result_path, "w"), indent=2)
