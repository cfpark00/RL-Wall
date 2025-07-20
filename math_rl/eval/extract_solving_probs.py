import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from math import comb

def estimate_pass_at_k(n, c, k):
    """
    Unbiased estimator for pass@k for a single problem.
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_at_k_batch(n_list, c_list, k):
    """
    Computes mean pass@k across multiple problems.

    Args:
        n_list (List[int]): Number of samples per problem.
        c_list (List[int]): Number of correct samples per problem.
        k (int): Number of attempts allowed.

    Returns:
        float: Mean pass@k across all problems.
    """
    assert len(n_list) == len(c_list)
    estimates = [
        estimate_pass_at_k(n, c, k)
        for n, c in zip(n_list, c_list)
    ]
    return sum(estimates) / len(estimates)

def plot_pass_at_k(k_values, pass_at_k_values, label=None, color="steelblue", save_path=None):
    # Set seaborn style
    sns.set_theme(style="whitegrid", context="talk")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot line with markers
    sns.lineplot(
        x=k_values,
        y=pass_at_k_values,
        marker="o",
        ax=ax,
        label=label,
        color=color,
        linewidth=2
    )
    # Highlight the point at k=64
    if 64 in k_values:
        idx_64 = k_values.index(64)
        x_64 = k_values[idx_64]
        y_64 = pass_at_k_values[idx_64]

        # Dashed guide lines
        ax.axvline(x=x_64, ymax=(y_64 - 0.58) / (1.02 - 0.58), color="red", linestyle="--", linewidth=1)
        ax.axhline(y=y_64, xmax=(x_64 - min(k_values)) / (max(k_values) - min(k_values)), color="red", linestyle="--", linewidth=1)

        # Optional: emphasize marker
        ax.plot(x_64, y_64, marker="o", color="red", markersize=7, markerfacecolor='red', zorder=10, label="Pass@64")

    # Axis labels and title
    ax.set_xlabel("k", fontsize=16, fontweight='bold')
    ax.set_ylabel("pass@k", fontsize=16, fontweight='bold')
    ax.set_ylim([0.58, 1.02])
    # ax.set_xscale("log")
    ax.tick_params(labelsize=13)

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    # Legend styling
    if label:
        ax.legend(fontsize=12, frameon=False)

    # Tight layout and save
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return 

def compute_solving_probs(json_file):
    """
    Reads a JSON file with a list of problem entries and computes solving probabilities
    based on 'corrects_verl_batched_responses' for each entry.

    Returns:
        A numpy array of solving probabilities (float values between 0 and 1)
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    probs = []
    for entry in data:
        responses = entry.get("corrects_verl_batched_responses", [])
        if responses:
            solving_prob = np.mean(responses)  # average of True/False
            probs.append(solving_prob)
        else:
            print('json file:', json_file)
            print(f"Warning: No responses found for entry {entry.keys()}.")
            probs.append(np.nan)  # handle empty/missing cases

    return np.array(probs)

def compute_avg_and_std_from_multiple_jsons(filepaths):
    """
    Given a list of JSON file paths (each containing a list of problem entries),
    compute the average and standard deviation of solving probabilities across seeds.

    Returns:
        mean_probs: numpy array of mean solving probabilities per problem
        std_probs:  numpy array of std deviation across seeds per problem
    """
    all_probs = []

    for path in filepaths:
        probs = compute_solving_probs(path)  # from previous function
        all_probs.append(probs)

    # Stack into 2D array: shape (num_seeds, num_problems)
    all_probs = np.vstack(all_probs)

    # Compute mean and std along the seed axis (axis=0)
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)

    return mean_probs, std_probs

def plot_solving_probs(pre_grpo_solve_probs, pre_grpo_std_probs, post_grpo_solve_probs, post_grpo_std_probs, image_name):
    """
    Plot the solving probabilities before and after GRPO.
    """
    # Calculate difference and error
    # Compute difference and propagated std
    delta = post_grpo_solve_probs - pre_grpo_solve_probs
    delta_std = np.sqrt(post_grpo_std_probs**2 + pre_grpo_std_probs**2)

    # Sort by pre-GRPO accuracy (descending)
    sort_idx = np.argsort(pre_grpo_solve_probs)[::-1]
    delta_sorted = delta[sort_idx]
    delta_std_sorted = delta_std[sort_idx]
    pre_sorted = pre_grpo_solve_probs[sort_idx]
    pre_std_sorted = pre_grpo_std_probs[sort_idx]

    x = np.arange(len(delta_sorted))

    # Plot setup
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['axes.grid'] = False
    fig, ax1 = plt.subplots(figsize=(6, 4.5))

    # Primary Y-axis: Δ Post - Pre
    ax1.errorbar(
        x, delta_sorted, yerr=delta_std_sorted,
        fmt='o', color='steelblue', ecolor='lightsteelblue',
        elinewidth=1, capsize=3, label="Δ Precision (Post - Pre)",
        markersize=2,
        zorder=0
    )
    # ax1.axhline(0, linestyle='--', color='black', linewidth=1, zorder=5)
    ax1.set_xlabel("Problem Index (sorted by Pre-GRPO precision)", fontsize=14)
    ax1.set_ylabel("Δ Precision", fontsize=14, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_ylim([-0.5, 1.0])
    ax1.set_xlim([0, 500])

    # Secondary Y-axis: Pre-GRPO accuracy with shaded error band
    ax2 = ax1.twinx()
    ax2.plot(x, pre_sorted, color='crimson', label="Pre-GRPO Pass@64", linewidth=2, zorder=0)
    ax2.fill_between(x, pre_sorted - pre_std_sorted, pre_sorted + pre_std_sorted,
                    color='crimson', alpha=0.2)
    ax2.set_ylabel("Pre-GRPO Precision", fontsize=14, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='crimson')

    
    if "train" in image_name:
        # guide lines
        ax1.vlines(305, 0.16, 0.48, color="forestgreen", linewidth=2, ls='--', zorder=5)
        ax1.hlines(0.16, 305, 510, color='crimson', linewidth=1, ls='--', zorder=5)
        ax1.hlines(0.47, 0, 305, color='steelblue', linewidth=1, ls='--', zorder=5)
        ax1.plot(305, 0.47, marker='*', markersize=14, color='forestgreen', zorder=0)
        ax1.axvspan(450, 500, color="gold", alpha=0.2)
    elif "test" in image_name:
        # guide lines
        ax1.vlines(250, 0.29, 0.37, color="darkorchid", linewidth=2, ls='--', zorder=5)
        ax1.hlines(0.29, 250, 510, color='crimson', linewidth=1, ls='--', zorder=5)
        ax1.hlines(0.37, 0, 250, color='steelblue', linewidth=1, ls='--', zorder=5)
        ax1.plot(250, 0.37, marker='*', markersize=14, color='darkorchid', zorder=0)
        ax1.axvspan(460, 500, color="gold", alpha=0.2)

    # Legends
    ax1.legend(loc='upper left', fontsize=11)
    ax2.legend(loc='lower left', fontsize=11)

    if "train" in image_name:
        plt.title("MATH (train subset)", fontsize=16)
    elif "test" in image_name:
        plt.title("MATH-500 (test)", fontsize=16)
    else:
        pass
    plt.tight_layout()
    plt.savefig(image_name, bbox_inches="tight")

    return 

if __name__ == "__main__":
    # model_name = "qwen-2.5-1.5b-instruct"
    # dataset_name = "math_500_train_subset"
    # pre_temp = "0.6"
    # post_temp = "1.0"
    # filepaths = [
    #     f"/n/home05/sqin/wall/verl/eval/data/{model_name}/{dataset_name}/temp={pre_temp}_seed={seed}/data.json"
    #     for seed in [1, 2, 3, 4]
    # ]
    # pre_grpo_solve_probs, pre_grpo_std_probs = compute_avg_and_std_from_multiple_jsons(filepaths)
    # # print(pre_grpo_solve_probs)

    # model_name = "qwen-1.5b-grpo-math-95steps"
    # filepaths = [
    #     f"/n/home05/sqin/wall/verl/eval/data/{model_name}/{dataset_name}/temp={post_temp}_seed={seed}/data.json"
    #     for seed in [1, 2, 3, 4]
    # ]
    # post_grpo_solve_probs, post_grpo_std_probs = compute_avg_and_std_from_multiple_jsons(filepaths)
    # # print(post_grpo_solve_probs)

    # # Plot the solving probabilities
    # plot_solving_probs(pre_grpo_solve_probs, pre_grpo_std_probs, 
    #                    post_grpo_solve_probs, post_grpo_std_probs, 
    #                    "train_solve_probs_delta.pdf"
    #                    )
    

    # model_name = "qwen-2.5-1.5b-instruct"
    # dataset_name = "math_500"
    # pre_temp = "0.6"
    # post_temp = "1.0"
    # filepaths = [
    #     f"/n/home05/sqin/wall/verl/eval/data/{model_name}/{dataset_name}/temp={pre_temp}_seed={seed}/data.json"
    #     for seed in [1, 2, 3, 4]
    # ]
    # pre_grpo_solve_probs, pre_grpo_std_probs = compute_avg_and_std_from_multiple_jsons(filepaths)
    # # print(pre_grpo_solve_probs)

    # model_name = "qwen-1.5b-grpo-math-95steps"
    # filepaths = [
    #     f"/n/home05/sqin/wall/verl/eval/data/{model_name}/{dataset_name}/temp={post_temp}_seed={seed}/data.json"
    #     for seed in [1, 2, 3, 4]
    # ]
    # post_grpo_solve_probs, post_grpo_std_probs = compute_avg_and_std_from_multiple_jsons(filepaths)
    # # print(post_grpo_solve_probs)

    # # Plot the solving probabilities
    # plot_solving_probs(pre_grpo_solve_probs, pre_grpo_std_probs, 
    #                    post_grpo_solve_probs, post_grpo_std_probs, 
    #                    "test_solve_probs_delta.pdf"
    #                    )
    
    # plot solving probs for different k
    model_name = "qwen-2.5-1.5b-instruct"
    seed = 0
    dataset_name = "math_500"
    n = 256
    temp= 0.6

    filename = f"/n/home05/sqin/wall/verl/eval/data/{model_name}/{dataset_name}/temp={temp}_seed={seed}_n={n}/data.json" 
    pass_256 = compute_solving_probs(filename)
    n_list = n * np.ones(len(pass_256), dtype=int)
    c_list = (pass_256 * n_list).astype(int)
    # estimate pass@k for different k values
    ks = [2, 4, 8, 16, 32, 64, 128, 256]
    pass_k = []
    for k in ks: 
        pass_k.append(compute_pass_at_k_batch(n_list, c_list, k=k))
    plot_pass_at_k(ks, pass_k, label="Qwen2.5-1.5B-Instruct", color="steelblue", save_path="pass_at_k_pre_grpo.pdf")


