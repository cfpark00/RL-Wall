import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from extract_solving_probs import compute_solving_probs, compute_avg_and_std_from_multiple_jsons
import matplotlib.gridspec as gridspec


def plot_solving_probs_temp_covg(model_names, temperatures, dataset_name, seed, image_name, model_n, train_data, eval_data):
    # get all the solving probs 
    alldata = {}
    for model_name in model_names:
        for temperature in temperatures:
            filepath = f"/n/home05/sqin/wall/verl/eval/data/{model_name}/{dataset_name}/temp={temperature}_seed={seed}/data.json"
            solving_probs = compute_solving_probs(filepath)
            alldata[f"{model_name}_T={temperature}"] = {
                "probs": solving_probs,
                "pass@64": np.mean(solving_probs>0.0),
            }
    
    # plotting
    # Set style
    sns.set_theme(style="whitegrid", context="talk")

    # Temperatures to plot
    n_temps = len(temperatures)

    # Prepare trimmed colormap (avoid white part at the top of 'hot')
    base_cmap = plt.get_cmap("magma_r")
    trimmed_colors = base_cmap(np.linspace(0.25, 1.0, n_temps))  # cut off the brightest

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 5, 5, 5])  # left panel is wider
 
    # Left column: metadata text
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")
    info_text = (
        f"Model: {model_n}\n"
        f"Train: {train_data}\n"
        f"Eval: {eval_data}"
    )

    ax_text.text(0, 0.5, info_text, fontsize=16, va='center', ha='left', linespacing=1.5)

    axes = [fig.add_subplot(gs[i]) for i in range(1, 4)]
    for ax, model in zip(axes, model_names):
        for i, T in enumerate(temperatures):
            solve_probs = alldata[f"{model}_T={T}"]['probs']
            x = np.arange(len(solve_probs))
            ax.plot(x, np.sort(solve_probs)[::-1], label=f"T={T}", color=trimmed_colors[i])

        ax.set_xlabel("Self-Sorting Index")
        if "GRPO" in model.upper() or "GSM8K" in model.upper():
            ax.set_title("Post-GRPO")
        else:
            ax.set_title("Pre-GRPO")
        # ax.set_title("Post-GRPO" if "GRPO" in model.upper() else "Pre-GRPO")
        ax.set_ylim([-0.02, 1.02])
        ax.tick_params(labelsize=12)

    axes[0].set_ylabel("Precision", fontsize=14)
    axes[1].legend(title="Temperature", fontsize=10)

    # on the third subplot, plot number of problems sovled with best-of-k sampling
    ax = axes[2]
    pre_grpo_num_solved = []
    post_grpo_num_solved = []   
    for i, T in enumerate(temperatures):
        num_solved = alldata[f"{model_names[0]}_T={T}"]["pass@64"]*len(solve_probs)
        pre_grpo_num_solved.append(num_solved)
        
        num_solved = alldata[f"{model_names[1]}_T={T}"]["pass@64"]*len(solve_probs)
        post_grpo_num_solved.append(num_solved)


    ax.plot(temperatures, pre_grpo_num_solved, color="skyblue", marker="o", label="Pre-GRPO")
    ax.plot(temperatures, post_grpo_num_solved, color="steelblue", marker="o", label="Post-GRPO")


    ax.axhline(y=np.max(pre_grpo_num_solved), color="skyblue", linestyle="--")
    ax.axhline(y=np.max(post_grpo_num_solved), color="steelblue", linestyle="--")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("# of Problems Solved")
    ax.set_ylim([None, len(solve_probs)])
    ax.set_xlim([-0.1, 1.3])
    ax.set_title("Pass@K, K=64")
    ax.legend(title="Model", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{image_name}.pdf", bbox_inches='tight')

    return

def compute_mean_and_std(acc_array):
    # Compute mean and standard deviation
    acc_array = acc_array * 500
    mean = np.mean(acc_array)
    std = np.std(acc_array)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std



if __name__ == "__main__":
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    model_names = ["llama-3.1-8b-instruct", "llama-3.1-8b-gsm8k"]
    dataset_name = "math_500"
    seed = 0
    image_name = 'solving_probs_temp_covg8b_llama_math'
    model = "Llama3.1-8B"
    train_data = "GSM8K"
    eval_data = "MATH-500"

    plot_solving_probs_temp_covg(model_names, temperatures, dataset_name, seed, image_name, model, train_data, eval_data)

    # acc_array = np.array([  0.886, 0.890, 0.884,  0.888, 0.892])
    # print(compute_mean_and_std(acc_array))