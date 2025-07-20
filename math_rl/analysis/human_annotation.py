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


def extract_options(text):
    converting_dict = {'YES': 1, 'NO': 0, 'N.A.': -1,  'N.A': -1}
    text = text.upper() #.replace("“", "").replace("”", "").replace("'", "").replace("'", "")
    text = text.replace("\n", " ")
    text = text.replace("*", "")
    text = text.replace("\BOXED{", "")
    text = text.replace("\TEXT{", "")
    text = text.replace("}", "")
    text = text.replace("(1)", "")
    text = text.replace("(2)", "")
    text = text.replace(",", "")
    text = text.replace("\;", "")
    text = text.replace(";", "")
    text = text.replace("(PARTIAL ATTEMPT)", "")
    
    # text = text.replace(" ", "_")
    # now just need to find "YES NO" or "NO N.A." or "YES YES."
    pattern = r"(YES|NO)\s*(YES|NO|N\.A)\s*"
    match = re.search(pattern, text)
    if match:
        return converting_dict[match.group(1)], converting_dict[match.group(2)]

    # === Pattern 1: \boxed{[\text{YES}, \text{N.A.}]}
    pattern_1 = r"""
        \[?\s*                    # optional [
        (?:\\TEXT\s*{)?\s*(YES|NO)\s*(?:})?\s*,\s*   # first answer
        (?:\\TEXT\s*{)?\s*(YES|NO|N\.A\.?)\s*(?:})?  # second answer
        \s*\]?\s*                 # optional ]
    """
    match = re.search(pattern_1, text, re.IGNORECASE | re.VERBOSE)
    if match:
        return converting_dict[match.group(1)], converting_dict[match.group(2)]

    # === Pattern 2: Plaintext (1) YES (2) N.A.
    pattern_2 = r"\(1\)\s*(YES|NO)\s*\(2\)\s*(YES|NO|N\.A\.?)"
    match = re.search(pattern_2, text, re.IGNORECASE | re.VERBOSE)
    if match:
        return converting_dict[match.group(1)], converting_dict[match.group(2)]
    
    # === Pattern 3: Plain [YES, NO] or [NO, N.A.]
    pattern_3 = r"\[\s*(YES|NO)\s*,\s*(YES|NO|N\.A\.?)\s*\]"
    match = re.search(pattern_3, text, re.IGNORECASE)
    if match:
        return converting_dict[match.group(1)], converting_dict[match.group(2)]
    
    if match:
        return converting_dict[match.group(1)], converting_dict[match.group(2)]
    
    else:
        print("No match found. Trying another pattern.")
    return None, None

def sanitize_for_verbatim(text):
    """
    Cleans LLM/Markdown-style text for use inside LaTeX verbatim environments:
    - Replaces smart quotes and long dashes with ASCII equivalents
    - Removes non-breaking spaces and problematic UTF-8 characters
    - Does NOT escape LaTeX special characters (intended for verbatim/minted environments)
    """
    replacements = {
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '–': '--', '—': '---',
        '…': '...',
        '\u00a0': ' ',    # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove control characters and enforce UTF-8 safe output
    text = text.encode('utf-8', 'replace').decode('utf-8', 'replace')

    return text

def gpt_idea_summary(args):
    # load the model response
    data = load_model_response(args.model_size, args.result_file)

    # manually inspect 500 test problems
    gt_idea_summary_all = {}
    for i in tqdm(range(len(data))):
    # for i in range(5):
        unique_id = data[i]['id']
        problem = data[i]['problem']
        gt_solution = data[i]['solution']

        instruction = f"Below is a math problem and it's solution trace: \n" \
            f"Problem: {problem} \n" \
            "Solution: " + str(gt_solution) + "\n" \
            f"Please use a few sentences to briefly describe the major steps required to solve this problem. "\
            f"Please do not include any mathematical details of the solution. "\
            f"The general steps should only outline the key steps required to correctly solve the problem. "\
            f"If the solution require consider different cases, please list these cases as well. Otherwise, no need to mention separate cases. "\
            f"Please start the reponse with: Here are the major steps required to solve this type of problem: \n"

        response = client.responses.create(
            model="gpt-4.1",
            input=instruction,
            temperature=0,
            # service_tier="flex",
        )

        idea_summary = response.output_text
        print("Idea Summary: ", idea_summary)
        gt_idea_summary_all[unique_id] = idea_summary
    
    # save results as json file
    result_dir = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}'
    gt_summary_file = os.path.join(result_dir, f'gt_idea_summary.json')
    with open(gt_summary_file, 'w') as f:
        json.dump(gt_idea_summary_all, f)
    print("Results saved to: ", gt_summary_file)
    
    return

def batch_request_json_creation(args):
    # load the model response
    data = load_model_response(args.model_size, args.result_file)
    # load idea summary
    gt_idea_summary_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/gt_idea_summary.json' 
    with open(gt_idea_summary_file, 'r') as f:
        gt_idea_summary_all = json.load(f)

    # create batch request to manually inspect 500 test problems
    all_requests = []
    for i in tqdm(range(len(data))):
        unique_id = data[i]['id']
        problem = data[i]['problem']
        gt_solution = data[i]['solution']
        gt_idea_summary = gt_idea_summary_all[unique_id]

        # grade student's solution
        grading_requests = []
        for j in range(len(data[i]['responses'])):
            grading_instruction = '''Now, please carefully read through the following solution and answer two questions below. \n 
            (1) Does the solution contain attempts to solve the problem using the approach described above? At this point, it is okay if the solution makes mathematical errors as long as the approach itself is correct. \n 
            (2) If the previous answer is yes, does the solution correctly execute required solution steps without making any critical math errors? Please use "N.A." if the answer to the previous question is "No". \n
            Feel free to elaborate but please provide a final answer using format: [YES/NO, YES/NO/N.A.].'''

            student_solution = data[i]['responses'][j]
            # print("j", "Student Solution: ", student_solution)
            
            instruction = f"You are a responsible grader and your task is to grade a competition-level high school math exam. \n" \
                "Below is a problem, it's solution trace and the general idea behind the solution: \n" \
                f"Problem: {problem} \n" \
                "Solution Trace: " + str(gt_solution) + "\n" \
                "Solution Idea: " + str(gt_idea_summary) + "\n" \
                + str(grading_instruction) + "\n" \
                + "Student Solution Attempt: " + str(student_solution) + "\n" 

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
    grading_request_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}/{args.result_file}_mini_grading_request.jsonl'
    with open(grading_request_file, "w") as f:
        for obj in all_requests:
            f.write(json.dumps(obj) + "\n")

    return

def upload_batch_request(args):
    grading_request_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}/{args.result_file}_mini_grading_request.jsonl'
    print("Uploading grading request file: ", grading_request_file)
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
            "description": "Grading Request",
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
    result_file = f'{args.result_file}_graded_mini.json'
    file_path = os.path.join(result_dir, result_file)
    with open(file_path, 'w') as f:
        f.write(batch_results.text)
    print("Results saved to: ", file_path)
    return


def summarize_results_model():
    result_dir = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}'
    if "temp=1.0_n=64" in args.result_file: 
        graded_files = ['pre_temp=1.0_n=64.json_graded_mini.json', 'post_temp=1.0_n=64.json_graded_mini.json']
    elif "temp=0.0_n=1" in args.result_file:
        graded_files = ['pre_temp=0.0_n=1.json_graded_mini.json', 'post_temp=0.0_n=1.json_graded_mini.json']
    else:
        raise ValueError("Invalid result file name. Please check the file name.")
    
    grading_response_files = []
    direction_grades_files = []
    execution_grades_files = []
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        print(file, "Number of graded responses: ", len(data))
        grading_response_all = {}
        for i in range(len(data)):
            graded = data[i]
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
            grading_id = graded["custom_id"]
            id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
            problem_id, response_id = int(id.group(1)), id.group(2)
            direction_grade, execution_grade = extract_options(grading_response)
            if direction_grade is None or execution_grade is None:
                continue
            if problem_id not in grading_response_all:
                grading_response_all[problem_id] = [[direction_grade, execution_grade]]
            else:
                grading_response_all[problem_id].append([direction_grade, execution_grade])
        grading_response_files.append(grading_response_all)
        
        avg_direction_grades_all = np.empty(500)
        avg_execution_grades_all = np.empty(500)
        avg_direction_grades_all[:] = np.nan
        avg_execution_grades_all[:] = np.nan
        for key, val in grading_response_all.items():
            # calculate the average of the two grades
            direction_grades = [x[0] for x in val]
            execution_grades = [x[1] for x in val if x[1] != -1]
            avg_direction_grades = sum(direction_grades) / len(direction_grades)
            avg_direction_grades_all[key] = avg_direction_grades
            if len(execution_grades) == 0:
                avg_execution_grades = np.nan
            else:
                avg_execution_grades = sum(execution_grades) / len(execution_grades)
            avg_execution_grades_all[key] = avg_execution_grades

        avg_direction_grades_all = np.array(avg_direction_grades_all)
        avg_execution_grades_all = np.array(avg_execution_grades_all)  
        # print(avg_direction_grades_all)
        # print(avg_execution_grades_all)

        direction_grades_files.append(avg_direction_grades_all)
        execution_grades_files.append(avg_execution_grades_all)
    
    direction_grades_files = np.array(direction_grades_files)
    execution_grades_files = np.array(execution_grades_files)
    # save as npy file
    if "temp=1.0_n=64" in args.result_file: 
        suffix='T=1.0_n=64'
    elif "temp=0.0_n=1" in args.result_file:
        suffix='T=0.0_n=1'
    else:
        raise ValueError("Invalid result file name. Please check the file name.")
    
    np.savez(f'{args.model_size}_execution_vs_direction_grades_{suffix}', 
             direction_grades_files=direction_grades_files, 
             execution_grades_files=execution_grades_files)
    
    return

def compare_models(temp=1.0):
    # Load data
    if temp == 1.0:
        results_0_5b = np.load('0.5b_execution_vs_direction_grades_T=1.0_n=64.npz')
        results_1_5b = np.load('1.5b_execution_vs_direction_grades_T=1.0_n=64.npz')
        results_7b = np.load('7b_execution_vs_direction_grades_T=1.0_n=64.npz')
        k = 64
    elif temp == 0.0:
        results_0_5b = np.load('0.5b_execution_vs_direction_grades_T=0.0_n=1.npz')
        results_1_5b = np.load('1.5b_execution_vs_direction_grades_T=0.0_n=1.npz')
        results_7b = np.load('7b_execution_vs_direction_grades_T=0.0_n=1.npz')
        k = 1

    # Extract pre/post
    dir_0_5b_pre = results_0_5b['direction_grades_files'][0]*100
    dir_0_5b_post = results_0_5b['direction_grades_files'][1]*100
    dir_1_5b_pre = results_1_5b['direction_grades_files'][0]*100
    dir_1_5b_post = results_1_5b['direction_grades_files'][1]*100
    dir_7b_pre = results_7b['direction_grades_files'][0]*100
    dir_7b_post = results_7b['direction_grades_files'][1]*100

    exe_0_5b_pre = results_0_5b['execution_grades_files'][0]*100
    exe_0_5b_post = results_0_5b['execution_grades_files'][1]*100
    exe_1_5b_pre = results_1_5b['execution_grades_files'][0]*100
    exe_1_5b_post = results_1_5b['execution_grades_files'][1]*100
    exe_7b_pre = results_7b['execution_grades_files'][0]*100
    exe_7b_post = results_7b['execution_grades_files'][1]*100

    # Compute means
    direction_means = [np.nanmean(dir_0_5b_pre), np.nanmean(dir_0_5b_post),
                       np.nanmean(dir_1_5b_pre), np.nanmean(dir_1_5b_post),
                       np.nanmean(dir_7b_pre), np.nanmean(dir_7b_post)]
    execution_means = [np.nanmean(exe_0_5b_pre), np.nanmean(exe_0_5b_post),
                       np.nanmean(exe_1_5b_pre), np.nanmean(exe_1_5b_post),
                       np.nanmean(exe_7b_pre), np.nanmean(exe_7b_post)]

    # Bar settings
    sns.set_theme(style="whitegrid", context="talk")
    sns.despine()
    model_labels = ['0.5B', '1.5B', '7B']
    x = np.arange(len(model_labels))
    bar_width = 0.25
    # chose colormap tab2 color
    pre_color = 'skyblue'
    post_color = 'steelblue'

    # Set up figure
    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    # --- Top: Direction ---
    ax = axes[0]
    ax.bar(x - bar_width/2, direction_means[::2], width=bar_width, color=pre_color, label='Pre-GRPO')
    ax.bar(x + bar_width/2, direction_means[1::2], width=bar_width, color=post_color, label='Post-GRPO')
    if temp == 1.0:
        ax.set_ylabel(f"Precision %", fontsize=14)
    else:
        ax.set_ylabel(f"Pass@{k}  %", fontsize=14)
    ax.set_title("Plan Grades", fontsize=15)
    ax.set_ylim(0, 100)
    

    # Annotate
    for i in range(len(x)):
        ax.text(x[i] - bar_width/2, direction_means[::2][i] + 0.01,
                f"{direction_means[::2][i]:.0f}%", ha='center', fontsize=11)
        ax.text(x[i] + bar_width/2, direction_means[1::2][i] + 0.01,
                f"{direction_means[1::2][i]:.0f}%", ha='center', fontsize=11)

    # --- Bottom: Execution ---
    ax = axes[1]
    ax.bar(x - bar_width/2, execution_means[::2], width=bar_width, color=pre_color, label='Pre-GRPO')
    ax.bar(x + bar_width/2, execution_means[1::2], width=bar_width, color=post_color, label='Post-GRPO')
    if temp == 1.0:
        ax.set_ylabel(f"Precision %", fontsize=14)
    else:
        ax.set_ylabel(f"Pass@{k}  %", fontsize=14)
    ax.set_title("Execution Grades", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)

    # Annotate
    for i in range(len(x)):
        ax.text(x[i] - bar_width/2, execution_means[::2][i] + 0.01,
                f"{execution_means[::2][i]:.0f}%", ha='center', fontsize=11)
        ax.text(x[i] + bar_width/2, execution_means[1::2][i] + 0.01,
                f"{execution_means[1::2][i]:.0f}%", ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"direction_execution_bar_pass{k}.pdf", bbox_inches="tight")

    return

def show_random_problems():
    results_1_5b = np.load('1_5b_execution_vs_direction_grades.npz')
    direction_grades_1_5b = results_1_5b['direction_grades_files']
    execution_grades_1_5b = results_1_5b['execution_grades_files']
    execution_pre = execution_grades_1_5b[0]
    execution_post = execution_grades_1_5b[1]

    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b'
    graded_files = ['pre_temp=1.0_n=64.json_graded_mini.json', 'post_temp=1.0_n=64.json_graded_mini.json']
    graded_response_files = []
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        graded_response_files.append(data)


    # load model responses
    pre_responses = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/pre_temp=1.0_n=64.json_mini_grading_request.jsonl'
    post_responses = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/post_temp=1.0_n=64.json_mini_grading_request.jsonl'
    # load the jsonl file
    with open(pre_responses, "r") as f:
        pre_response_data = []
        for line in f:
            pre_response_data.append(json.loads(line))
    with open(post_responses, "r") as f:
        post_response_data = []
        for line in f:
            post_response_data.append(json.loads(line))
        
    # randomly select a problem, and a response
    problem_id = np.random.randint(0, 500)
    response_id = np.random.randint(0, 64)

    # first, read the question and solutoin summary 
    for question in pre_response_data:
        if question["custom_id"].startswith(f"1.5b_pre_temp=1.0_n=64.json-{problem_id}-"):
            question = question["body"]["input"]

    # now examine the pre GRPO response
    for graded in graded_response_files[0]:
        if graded["custom_id"].startswith(f"1.5b_pre_temp=1.0_n=64.json-{problem_id}-{response_id}"):
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]

            grading_id = graded["custom_id"]
            id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
            problem_id, response_id = int(id.group(1)), int(id.group(2))
            direction_grade, execution_grade = extract_options(grading_response)
            if direction_grade is None or execution_grade is None:
                raise ValueError("Invalid grading response. Please try again.")
            print("Direction Grade: ", direction_grade)
            print("Execution Grade: ", execution_grade)
            
            
            print("-"*100)
            
            print(f"Grading request: \n", sanitize_for_verbatim(pre_response_data[problem_id*64 + response_id]["body"]["input"]))
            print("-"*20)
            print("Pre GRPO Grading Response: \n ", sanitize_for_verbatim(grading_response))
            break

    return    

def estimate_api_cost():
    enc = tiktoken.encoding_for_model("gpt-4")

    # average tokens for prompt:
    problem_data_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/pre_temp=1.0_n=64.json_new_grading_request.jsonl'
    # load the jsonl file
    with open(problem_data_file, "r") as f:
        question_data = []
        for line in f:
            question_data.append(json.loads(line))
    prompt_len_count = []
    for question in question_data:
        question = question["body"]["input"]
        prompt_len_count.append(len(enc.encode(question)))
    print("Average tokens for prompt: ", np.mean(prompt_len_count))
    print("Total number of requests: ", len(question_data))

    # get average tokens for response:
    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b'
    graded_files = ['pre_temp=1.0_n=64.json_graded_new.json']
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        print(file, "Number of graded responses: ", len(data))
        resp_len_count = []
        for i in range(15304): 
            graded = data[i]
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
            
            resp_len_count.append(len(enc.encode(grading_response)))
        print("Average tokens per response: ", np.mean(resp_len_count))
    
    # compute the cost
    # input tokens
    input_tokens = np.mean(prompt_len_count) * len(question_data)
    input_price = input_tokens / 1_000_000 * 0.2 # $1.00 per 1M tokens
    # output tokens
    output_tokens = np.mean(resp_len_count) * len(question_data)
    output_price = output_tokens / 1_000_000 * 0.8 # $4.00 per 1M tokens
    # total cost
    total_cost = input_price + output_price
    print("Total cost: ", total_cost)
    
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Annotation Script")

    parser.add_argument("--model_size", type=str, default="1.5b", 
                        help="0.5b, 1.5b or 7b")
    parser.add_argument("--result_file", type=str, default="pre_temp=1.0_n=64.json", 
                        help="/n/netscratch/dam_lab/Everyone/wall/cfpark00/for_manual_inspection")
    
    args = parser.parse_args()
    
    # gpt_idea_summary(args)

    # estimate_api_cost()

    # batch_request_json_creation(args)

    # upload_batch_request(args)
    
    # submit_batch_request(batch_input_file_id = post_grpo_batch_input_file_id)

    # check_status(batch_id=post_grpo_batch_id)

    # retrieve_results(batch_id=post_grpo_output_file_id)

    # summarize_results_model()

    # compare_models(temp=1.0)

    show_random_problems()
