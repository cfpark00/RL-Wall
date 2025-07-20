import argparse
import os, sys
import json
import re 
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import seaborn as sns
from human_annotation import extract_options
from matplotlib.ticker import FuncFormatter
from human_annotation import sanitize_for_verbatim

client = OpenAI()

def load_model_response(model_size, result_file):
    reponse_dir = f'/n/netscratch/dam_lab/Everyone/wall/cfpark00/for_manual_inspection/{model_size}'
    result_path = os.path.join(reponse_dir, result_file)
    # check if the file exists
    if not os.path.exists(result_path):
        reponse_dir = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{model_size}'
        result_path = os.path.join(reponse_dir, result_file)
    # load json file
    with open(result_path, 'r') as f:  
        data = json.load(f)
    print(data[0].keys())
    # data contains a list of dictionaries for each problem 
    # dict keys: 'problem', 'solution', 'answer', 'subject', 'level', 'id', 'messages', 'prompt', 'responses', 'corrects_verl_batched_responses', 'verl_model_answers'
    return data


def extract_options_rubric(text):
    """
    Extracts a 3-tuple from strings like:
    \boxed{YES, ELEMENTARY, YES}

    Returns:
        Tuple of (first, second, third), or (None, None, None) if no match.
    """
    text = text.upper().replace("\n", " ").strip()

    pattern = r"""
        \\BOXED\s*{              # match \boxed{ with optional whitespace
        \s*(YES|NO)\s*,          # first value
        \s*(ELEMENTARY|HIGH|NA)\s*,  # second value
        \s*(YES|NO)\s*           # third value
        }                        # closing }
    """

    match = re.search(pattern, text, re.IGNORECASE | re.VERBOSE)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        return None, None, None
    # extract options 

def load_execution_grade(args):
    result_dir = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}'
    result_file = f"{args.result_file}_graded_mini.json"

    file_path = os.path.join(result_dir, result_file)
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    print(file_path, "Number of graded responses: ", len(data))

    grading_response_all = {}
    for i in range(len(data)):
        graded = data[i]
        grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
        grading_id = graded["custom_id"]
        id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
        problem_id, response_id = int(id.group(1)), int(id.group(2))
        direction_grade, execution_grade = extract_options(grading_response)
        if direction_grade is None or execution_grade is None:
            direction_grade, execution_grade = -1, -1
        if problem_id not in grading_response_all:
            grading_response_all[problem_id] = -1 * np.ones(64)  
        grading_response_all[problem_id][response_id] = execution_grade

    return grading_response_all
    

def batch_request_json_creation(args):
    # load the model response
    data = load_model_response(args.model_size, args.result_file)
    execution_filter = load_execution_grade(args)

    # create batch request to manually inspect 500 test problems
    all_requests = []
    for i in tqdm(range(len(data))):
        unique_id = data[i]['id']
        problem = data[i]['problem']

        # grade student's solution
        grading_requests = []
        for j in range(len(data[i]['responses'])):
            # determine if the response is worth grading
            execution_correct = execution_filter[i][j]
            if execution_correct == -1 or execution_correct == 1: # either invalid or correct
                continue
            # alright, let's grade the response
            grading_rubric = '''
(1). Does the solution contain basic mathematical factual mistakes? Here we are talking about simple plain mathematical facts that are in direct contradiction to well-known mathematical knowledge. Examples of mathematical facts are: how many sides does a triangle have, is 2 an odd number. Note that if the students make a computational or algebraic mistakes when manipulating mathematical operations, we DO NOT count this as a factual mistake. We are looking for mistakes that can be spotted even without the problem or solution context. In other words, the mistake is obvious in isolation. \n
(2). If the answer to (1) is yes, is the math factual mistake an elementary (including middle school) one or a high-school one? Please output "ELEMENTARY" or "HIGH". If the answer to (2) is no, please output "NA". \n
(3). Does the solution contain basic logic mistakes (no math involved to spot such an error) that are obviously non-sensical? We are talking about very basic logic mistakes that even a human with the no mathematical knowledge can identify. To reiterate, we are looking for plain simple logic errors. For example, the student self-contradicting what was said before, or making a conclusion without clear logic from existing steps. Again, making mistakes in complex math concepts does not count here. \n
                            '''
            student_solution = data[i]['responses'][j]
            
            instruction = r"You are grading a high school competition exam. Below is the grading rubric. I will provide you the question and the student's solution attempt. Please make your best judgements on three grading criteria below. Feel free to elaborate but please output your final grading in the format of \boxed{YES/NO, ELEMENTARY/HIGH/NA, YES/NO}. It is very important that you follow the exact format in verbatim for the final grade. Please do not add any additional formatting in your final answer. \n" \
                f"Grading Rubric: {grading_rubric} \n " \
                f"Problem: \n {problem} \n " \
                "Solution Attempt: \n " + str(student_solution) + "\n " 

            # create dict entry file for batch request
            request_entry = {
                "custom_id": f"{args.model_size}_{args.result_file}-{i}-{j}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-4.1-mini",
                    "input": instruction,
                    "temperature": 0,
                    }
            }
            grading_requests.append(request_entry)
            
            all_requests.append(request_entry)
    
    # now dump the requests to the file
    grading_request_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}/{args.result_file}_mini_rubric_request.jsonl'
    with open(grading_request_file, "w") as f:
        for obj in all_requests:
            f.write(json.dumps(obj) + "\n")
    print(f"Batch request file created at {grading_request_file}")  

    return

def upload_batch_request(args):
    grading_request_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}/{args.result_file}_mini_rubric_request.jsonl'
    print("Uploading rubric request file: ", grading_request_file)
    batch_input_file = client.files.create(
        file=open(grading_request_file, "rb"),
        purpose="batch"
    )

    print(batch_input_file)
    return 

def submit_batch_request(batch_input_file_id):
    print("Submitting grading request file")
    # batch_input_file_id = batch_input_file.id
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": "Rubric Request",
        }
    )
    print("Batch request submitted.")
    print(batch_info)
    return

def check_status(batch_id):
    # Check the status of the batch
    batch_info = client.batches.retrieve(batch_id)
    print("Batch Status: ", batch_info)
    return

def retrieve_results(batch_id):
    # Retrieve the results of the batch
    batch_results = client.files.content(batch_id)
    # save as a jsonl file
    result_dir = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}'
    result_file = f'{args.result_file}_rubric_graded_mini.json'
    file_path = os.path.join(result_dir, result_file)
    with open(file_path, 'w') as f:
        f.write(batch_results.text)
    print("Results saved to: ", file_path)
    return

def summarize_results_model(args):
    result_dir = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}'
    result_file = f'{args.result_file}_rubric_graded_mini.json'
   
    file_path = os.path.join(result_dir, result_file)
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(result_file, "Number of graded responses: ", len(data))
    
    fact_all = []
    fact_level_all = []
    logic_all = []
    fmap = {
        "YES": 1,
        "NO": 0,
        "ELEMENTARY": 1,
        "HIGH": 2,
        "NA": 0,
    }
    for i in range(len(data)):
        graded = data[i]
        grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
        grading_id = graded["custom_id"]
        id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
        problem_id, response_id = int(id.group(1)), int(id.group(2))
        fact, fact_level, logic = extract_options_rubric(grading_response)
        if fact is None or fact_level is None or logic is None:
            fact, fact_level, logic = -1, -1, -1
            print("Error: ", grading_response)
        else:
            fact = fmap[fact]
            fact_level = fmap[fact_level]
            logic = fmap[logic]
        fact_all.append(fact)
        fact_level_all.append(fact_level)
        logic_all.append(logic)
    # convert to numpy array
    fact_all = np.array(fact_all)
    fact_level_all = np.array(fact_level_all)
    logic_all = np.array(logic_all)
    print("Fact grading: ", len(fact_all))
    print("Fact level grading: ", len(fact_level_all))
    print("Logic grading: ", len(logic_all))

    # save the results
    if "pre" in args.result_file:
        suffix = "pre"
    elif "post" in args.result_file:
        suffix = "post"
    else:
        suffix = "none"
    # save the results
    np.savez(f'{args.model_size}_rubric_grades_{suffix}_T=1.0_n=64', 
            fact_all=fact_all, 
            fact_level_all=fact_level_all, 
            logic_all=logic_all)

    return 

def compare_models(args):
    # read the pre and post npz files
    pre_file = f'1.5b_rubric_grades_pre_T=1.0_n=64.npz'
    post_file = f'1.5b_rubric_grades_post_T=1.0_n=64.npz'
    pre_data = np.load(pre_file)
    post_data = np.load(post_file)
    # also load the 7b pre
    pre_7b_file = f'7b_rubric_grades_pre_T=1.0_n=64.npz'
    post_7b_file = f'7b_rubric_grades_post_T=1.0_n=64.npz'
    pre_7b_data = np.load(pre_7b_file)
    post_7b_data = np.load(post_7b_file)
    
    # compare the two models
    # first, count number of lens
    pre_lens = len(pre_data["fact_all"] == 1)
    post_lens = len(post_data["fact_all"] == 1)
    pre_7b_lens = len(pre_7b_data["fact_all"] == 1)
    post_7b_lens = len(post_7b_data["fact_all"] == 1)
    print(f"1.5b Lens count: {pre_lens} vs {post_lens}")
    print(f"7b Lens count: {pre_7b_lens} vs {post_7b_lens}")

    # first, count number of fact grades
    pre_fact_count = np.sum(pre_data["fact_all"] == 1)
    post_fact_count = np.sum(post_data["fact_all"] == 1)
    pre_7b_fact_count = np.sum(pre_7b_data["fact_all"] == 1)
    post_7b_fact_count = np.sum(post_7b_data["fact_all"] == 1)
    print(f"1.5b Fact count: {pre_fact_count} vs {post_fact_count}")
    print(f"7b Fact count: {pre_7b_fact_count} vs {post_7b_fact_count}")
          
    # second, count number of fact level grades
    pre_fact_level_count = np.sum(pre_data["fact_level_all"] == 1)
    post_fact_level_count = np.sum(post_data["fact_level_all"] == 1)
    pre_7b_fact_level_count = np.sum(pre_7b_data["fact_level_all"] == 1)
    post_7b_fact_level_count = np.sum(post_7b_data["fact_level_all"] == 1)
    print(f"1.5b Fact ELEMENTARY level count: {pre_fact_level_count} vs {post_fact_level_count}")
    print(f"7b Fact ELEMENTARY level count: {pre_7b_fact_level_count} vs {post_7b_fact_level_count}")

    pre_fact_level_count = np.sum(pre_data["fact_level_all"] == 2)
    post_fact_level_count = np.sum(post_data["fact_level_all"] == 2)
    pre_7b_fact_level_count = np.sum(pre_7b_data["fact_level_all"] == 2)
    post_7b_fact_level_count = np.sum(post_7b_data["fact_level_all"] == 2)
    print(f"1.5b Fact level HIGH count: {pre_fact_level_count} vs {post_fact_level_count}")
    print(f"7b Fact level HIGH count: {pre_7b_fact_level_count} vs {post_7b_fact_level_count}")

    # third, count number of logic grades
    pre_logic_count = np.sum(pre_data["logic_all"] == 1)
    post_logic_count = np.sum(post_data["logic_all"] == 1)
    pre_7b_logic_count = np.sum(pre_7b_data["logic_all"] == 1)
    post_7b_logic_count = np.sum(post_7b_data["logic_all"] == 1)
    print(f"1.5b Logic count: {pre_logic_count} vs {post_logic_count}, 7b: {pre_7b_logic_count}")
    print(f"7b Logic count: {pre_7b_logic_count} vs {post_7b_logic_count}")


    return 

def plot_mistake_drops():
    sns.set_theme(style="whitegrid", context="talk")

    categories = [
        "Execution:\nAll Mistakes",
        "Factual:\nAll Mistakes",
        "Factual:\nElementary-Level",
        "Factual:\nHigh School-Level",
        "Basic Logic Mistakes"
    ]
    
    # colors
    cmap = plt.get_cmap("tab20b")
    purple_shades = [cmap(i) for i in range(16, 20)] 

    # Core data
    pre_1_5b = np.array([13437., 8801., 7572., 1230., 6266.])
    pre_1_5b_norm = np.array([1, 1, 1, 1, 1])
    post_1_5b = np.array([10792., 5784., 4669., 1116., 4194.])
    # normalize post with pre by performing element-wise division
    post_1_5b = np.divide(post_1_5b, pre_1_5b, out=np.zeros_like(post_1_5b), where=pre_1_5b != 0)

    pre_7b = np.array([8020., 3192., 2318., 881., 2304.])
    pre_7b = np.divide(pre_7b, pre_1_5b, out=np.zeros_like(post_1_5b), where=pre_1_5b != 0)
    # post_7b = [6988, 2741, 1951, 796, 2274]  # dummy values for now


    y_pos = list(range(len(categories)))
    y_offset = 0.

    fig, ax = plt.subplots(figsize=(4, 4))

    # === Plot 1.5B ===
    ax.scatter(pre_1_5b_norm, [y + y_offset for y in y_pos], color=purple_shades[3], alpha=1.0, label="1.5B Pre-GRPO", s=100, zorder=3)
    ax.scatter(post_1_5b, [y + y_offset for y in y_pos], color=purple_shades[2], alpha=1.0, label="1.5B Post-GRPO", s=100, zorder=3)
    for x0, x1, y in zip(pre_1_5b_norm, post_1_5b, y_pos):
        ax.annotate('', xy=(x1, y + y_offset), xytext=(x0, y + y_offset),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=3), zorder=2)

    # === Plot 7B ===
    ax.scatter(pre_7b, [y - y_offset for y in y_pos], color=purple_shades[1], alpha=1.0, label="7B Reference", s=100, zorder=3)
    # ax.scatter(post_7b, [y - y_offset for y in y_pos], color="tab:orange", alpha=1.0, label="7B Post-GRPO", s=100, zorder=3)
    # for x0, x1, y in zip(pre_7b, post_7b, y_pos):
    #     ax.annotate('', xy=(x1, y - y_offset), xytext=(x0, y - y_offset),
    #                 arrowprops=dict(arrowstyle='->', color='gray', lw=3), zorder=2)

    # === Labels and styling ===
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Number of Mistakes", fontsize=14)
    # ax.set_title("Mistake Reduction by Model Size & GRPO", fontsize=15)
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(-0.6, -0.02), frameon=False)

    # Format x-axis to "k" style
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))

    # plt.tight_layout()
    plt.savefig("execution_mistake_drops.pdf", bbox_inches='tight', dpi=300)

    return

def show_random_problems():
    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b'
    graded_files = ['pre_temp=1.0_n=64.json_rubric_graded_mini.json', 'post_temp=1.0_n=64.json_rubric_graded_mini.json']
    graded_response_files = []
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        graded_response_files.append(data)


    # load model responses
    pre_responses = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/pre_temp=1.0_n=64.json_mini_rubric_request.jsonl'
    post_responses = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/post_temp=1.0_n=64.json_mini_rubric_request.jsonl'
    # load the jsonl file
    with open(pre_responses, "r") as f:
        pre_response_data = []
        for line in f:
            pre_response_data.append(json.loads(line))
    with open(post_responses, "r") as f:
        post_response_data = []
        for line in f:
            post_response_data.append(json.loads(line))

    # randomly select a grading request
    request_id = np.random.randint(0, len(pre_response_data))
    # get the response
    
    request = pre_response_data[request_id]
    grading_response = graded_response_files[0][request_id]
    # first check the custom_id match
    assert request["custom_id"] == grading_response["custom_id"], "Custom ID does not match"
    grading_response_text = grading_response["response"]["body"]["output"][0]["content"][0]["text"]

    # now print the request
    fact, fact_level, logic = extract_options_rubric(grading_response_text)
    if fact is None or logic is None:
        raise ValueError("Invalid grading response. Please try again.")
    print("Fact Grade: ", fact)
    print("fact_level: ", fact_level)
    print("Logic Grade: ", logic)
    
    print("-"*100)
    
    print(f"Grading request: \n", sanitize_for_verbatim(request["body"]["input"]))
    print("-"*20)
    print("Pre GRPO Grading Response: \n ", sanitize_for_verbatim(grading_response_text))


    return    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Annotation Script")

    parser.add_argument("--model_size", type=str, default="1.5b", 
                        help="0.5b, 1.5b or 7b")
    parser.add_argument("--result_file", type=str, default="pre_temp=1.0_n=64.json", 
                        help="/n/netscratch/dam_lab/Everyone/wall/cfpark00/for_manual_inspection")
    
    args = parser.parse_args()
    
    # batch_request_json_creation(args)

    # upload_batch_request(args)
    
    # pre_grpo_batch_input_file_id = 'file-BaJeDyT6gSEKQYE2h9vjKo'
    # post_grpo_batch_input_file_id = 'file-EZTjehtMxJYTr2EZwvBDF3' 
    # pre_grpo_batch_input_file_id = 'file-WvbguudSLJeCKqH68kpcEa' 
    # post_grpo_batch_input_file_id = 'file-V1MbVbErBEXvwKvR6VTuDZ'
    # submit_batch_request(batch_input_file_id = post_grpo_batch_input_file_id)

    # pre_grpo_batch_id = 'batch_68224f4426b4819080e46b2d2b2bdbed' 
    # post_grpo_batch_id = 'batch_68224f6854508190b9ca8e9e01b7b6fc' 
    # pre_grpo_batch_id = 'batch_682251b62bfc81909f9673f92a762fd0'
    # post_grpo_batch_id = 'batch_682289c0878081908b16a3f32ef1cc5d' 
    # check_status(batch_id=post_grpo_batch_id)

    # pre_grpo_batch_output_file_id = 'file-9ujRs7xZvtYzzzzNn4nwgq'
    # post_grpo_batch_output_file_id = 'file-1xPsjaZEgp17GfxMRw76oK'
    # pre_grpo_batch_output_file_id = 'file-TtQxgFvfvW1Faw4dXZ9RW1'
    # post_grpo_batch_output_file_id = 'file-4HDz9yE72qN9HinAkkus7f'
    # retrieve_results(batch_id=post_grpo_batch_output_file_id)

    # summarize_results_model(args)

    # compare_models(args)

    # plot_mistake_drops()

    show_random_problems()



    
