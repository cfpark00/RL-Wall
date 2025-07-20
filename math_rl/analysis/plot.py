import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

'''
    data info:
        corrects: array of size (500, 64), indicating whether each guess is correct
        probs: array of of size (500,) indicating the pass@64 rate for each sample
        lower: ???
        upper: ???
        n_corretcts:
        n_trials:
        pass@k: one floating number, proportions of problems covered with "best-of-k" sampling    
'''
# load data
alldata = json.load(open("alldata.json", "r"))
Ts = [0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
print(alldata.keys())

def acc_vs_steps():
    # steps
    pass_1_0 = [alldata[f"Qwen-1.5B_GRPO_{step}steps_T=0"]["pass@1"] for step in steps]
    pass_1 = [alldata[f"Qwen-1.5B_GRPO_{step}steps_T=1.0"]["pass@1"] for step in steps]
    pass_32 = [
        alldata[f"Qwen-1.5B_GRPO_{step}steps_T=1.0"]["pass@32"] for step in steps
    ]

    plt.plot(steps, pass_1_0, marker="o", label="Qwen-1.5B_GRPO_T=0 pass@1")
    plt.plot(steps, pass_1, marker="o", label="Qwen-1.5B_GRPO_T=1.0 pass@1")
    plt.plot(steps, pass_32, marker="o", label="Qwen-1.5B_GRPO_T=1.0 pass@32")
    plt.xticks(steps)
    plt.xlabel("Steps")
    plt.ylabel("Acc")
    plt.legend()
    plt.savefig("acc_vs_steps.png", dpi=300)

def acc_vs_temp():
    ''' deprecated '''
    n_samples = [1, 2, 4, 8, 16, 32, 64]
    cmap = plt.get_cmap("jet_r")
    n_sample_colors = {}
    for i_n_sample, n_sample in enumerate(n_samples):
        n_sample_colors[n_sample] = cmap(i_n_sample / len(n_samples))

    passes = []
    passes_rl = []
    for T in Ts:
        fn = f"Qwen-1.5B_T={T}"
        fn_rl = f"Qwen-1.5B_GRPO_95steps_T={T}"
        passes_T, passes_rl_T = [], []
        for i_n_sample, n_sample in enumerate(n_samples):
            key = f"pass@{n_sample}"
            if key in alldata[fn]:
                passes_T.append(alldata[fn][key])
            else:
                passes_T.append(np.nan)
            if key in alldata[fn_rl]:
                passes_rl_T.append(alldata[fn_rl][key])
            else:
                passes_rl_T.append(np.nan)
        passes.append(passes_T)
        passes_rl.append(passes_rl_T)
    passes = np.array(passes)
    passes_rl = np.array(passes_rl)
    passes.shape, passes_rl.shape

    plt.figure()
    for i_n_sample, n_sample in enumerate(n_samples):
        plt.plot(Ts, passes[:, i_n_sample], color=n_sample_colors[n_sample], ls="--")
        plt.plot(Ts, passes_rl[:, i_n_sample], color=n_sample_colors[n_sample], ls="-")
    for i_n_sample, n_sample in enumerate(n_samples):
        plt.plot(
            [],
            [],
            label="n_sample={}".format(n_sample),
            color=n_sample_colors[n_sample],
        )
    plt.plot([], [], ls="--", label="Pre-GRPO", c="black")
    plt.plot([], [], ls="-", label="Post-GRPO", c="black")
    plt.xlabel("Temperature")
    plt.ylabel("Pass")
    plt.legend(ncol=2)
    plt.xscale("log")
    plt.savefig("acc_vs_T.png", dpi=300)

    return

def solving_probs_temp():
    # Set global style
    sns.set_theme(style="whitegrid", context="talk")

    # Temperatures to plot
    temperatures = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    n_temps = len(temperatures)

    # Prepare trimmed colormap
    base_cmap = plt.get_cmap("magma_r")
    trimmed_colors = base_cmap(np.linspace(0.25, 1.0, n_temps))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    model_names = ["Qwen-1.5B", "Qwen-1.5B_GRPO_95steps"]

    for ax, model in zip(axes[:2], model_names):
        for i, T in enumerate(temperatures):
            solve_probs = alldata[f"{model}_T={T}"]["probs"]
            x = np.arange(len(solve_probs))
            ax.plot(x, np.sort(solve_probs)[::-1], label=f"T={T}", color=trimmed_colors[i], linewidth=2)

        ax.set_xlabel("Self-Sorting Rank", fontsize=16)
        ax.set_title("Post-GRPO" if "GRPO" in model else "Pre-GRPO", fontsize=16, fontweight='bold')
        ax.set_ylim([-0.02, 1.02])
        ax.tick_params(labelsize=13)

        # Customize spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

    axes[0].set_ylabel("Precision", fontsize=16)
    axes[1].legend(title="Temperature", fontsize=11, title_fontsize=14)

    # Third subplot: Pass@64
    ax = axes[2]
    pre_grpo_num_solved = []
    post_grpo_num_solved = []
    for i, T in enumerate(temperatures):
        pre = alldata[f"Qwen-1.5B_T={T}"]["pass@64"] * 500
        post = alldata[f"Qwen-1.5B_GRPO_95steps_T={T}"]["pass@64"] * 500
        pre_grpo_num_solved.append(pre)
        post_grpo_num_solved.append(post)

    pre_grpo_std = np.array([0, 0.2, 0.4, 1.3, 2.5, 4.48, 2.0, 1.87, 4.75])
    post_grpo_std = np.array([0, 0.2, 0.6, 4.2, 1.8, 3.3, 2.5, 2.28, 1.4])
    # ax.plot(temperatures, pre_grpo_num_solved, marker="o", color="skyblue", label="Pre-GRPO", linewidth=2)
    # ax.plot(temperatures, post_grpo_num_solved, marker="o", color="steelblue", label="Post-GRPO", linewidth=2)
    ax.errorbar(temperatures, pre_grpo_num_solved, yerr=pre_grpo_std, marker="o", linestyle="-", color="skyblue", label="Pre-GRPO", capsize=4, linewidth=2, markersize=5)
    ax.errorbar(temperatures, post_grpo_num_solved, yerr=post_grpo_std, marker="o", linestyle="-", color="steelblue", label="Post-GRPO", capsize=4, linewidth=2, markersize=5)
    ax.axhline(y=np.max(pre_grpo_num_solved), color="skyblue", linestyle="--", linewidth=1.5)
    ax.axhline(y=np.max(post_grpo_num_solved), color="steelblue", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Temperature", fontsize=16)
    ax.set_ylabel("# of Problems Solved", fontsize=16)
    ax.set_ylim([None, 500])
    ax.set_title("Pass@K, K=64", fontsize=16, fontweight='bold')
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=13)

    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    plt.tight_layout()
    plt.savefig("solve_probs.pdf", dpi=300, bbox_inches='tight')

    return

def solving_probs_matched():
    # Set style
    sns.set_theme(style="whitegrid", context="talk")
    T = 1.0

    # Create figure with 2 shared-y subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    pre_grpo_solve_probs = np.array(alldata[f"Qwen-1.5B_T={T}"]["probs"])
    post_grpo_solve_probs = np.array(alldata[f"Qwen-1.5B_GRPO_95steps_T={T}"]["probs"])
    x = np.arange(len(pre_grpo_solve_probs))
    sort_idx = np.argsort(pre_grpo_solve_probs)[::-1]

    ax.plot(x, pre_grpo_solve_probs[sort_idx], color='C0', label="Pre-GRPO", ls="", marker="o", markersize=5)
    ax.plot(x, post_grpo_solve_probs[sort_idx], color='C1', label="Post-GRPO", ls="", marker="o", markersize=5)

    ax.set_xlabel("Test Problems")
    ax.set_ylim([-0.02, 1.02])
    ax.tick_params(labelsize=12)
    ax.set_ylabel("Pass@64 Rate", fontsize=14)
    plt.legend(title='T=1.0', fontsize=12)
    plt.savefig("solve_probs_matched.pdf", bbox_inches="tight")

    return

def subject_vs_acc():
    annot = True
    probs_base = np.array(alldata["Qwen-1.5B_T=1.0"]["probs"])
    subjects = list(alldata["subject_inds"].keys()) + ["all"]

    # Load GRPO accuracies
    probs_grpo_subjects = np.array(
        [np.array(alldata[f"Qwen-1.5B_GRPO_{subject}_T=1.0"]["probs"]) for subject in subjects]
    )

    # Compute subject-wise means
    base_probs_per_subjects = []
    probs_per_subjects = []
    for key, subject_ind in alldata["subject_inds"].items():
        base_probs_per_subjects.append(probs_base[subject_ind].mean())
        probs_per_subjects.append(probs_grpo_subjects[:, subject_ind].mean(axis=-1))
    base_probs_per_subjects = np.array(base_probs_per_subjects)
    probs_per_subjects = np.array(probs_per_subjects)  # [n_subjects, n_subjects+1]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # === Left: Accuracy Matrix (green = good, red = bad) ===
    sns.heatmap(
        probs_per_subjects,
        xticklabels=subjects,
        yticklabels=list(alldata["subject_inds"].keys()),
        cmap="RdYlGn",
        center=np.mean(probs_per_subjects),
        cbar_kws={'label': 'Precision', 'shrink': 0.9},
        ax=axs[0],
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        vmin=0.0,
        vmax=1.0
    )
    axs[0].set_xlabel("Training Subject", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("Test Subject", fontsize=14, fontweight='bold')
    axs[0].tick_params(labelsize=12, rotation=45)
    axs[0].set_title("GRPO Precision Matrix", fontsize=15, fontweight='bold')
    cbar = axs[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)  # adjust tick font size
    cbar.set_label("Precision", fontsize=14, fontweight='bold')  # label font siz

    # === Right: Δ Accuracy Matrix (centered around 0, higher contrast) ===
    delta = probs_per_subjects - base_probs_per_subjects[:, None]
    max_abs_delta = np.max(np.abs(delta))
    sns.heatmap(
        delta,
        xticklabels=subjects,
        yticklabels=list(alldata["subject_inds"].keys()),
        cmap="seismic",  # high-contrast diverging
        center=np.mean(delta),
        cbar_kws={'label': 'Δ Precision', 'shrink': 0.9},
        ax=axs[1],
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        vmin=-max_abs_delta,
        vmax=max_abs_delta
    )
    axs[1].set_xlabel("Training Subject", fontsize=14, fontweight='bold')
    axs[1].set_ylabel("Test Subject", fontsize=14, fontweight='bold')
    axs[1].tick_params(labelsize=12, rotation=45)
    axs[1].set_title("Improvement over Baseline", fontsize=15, fontweight='bold')
    cbar = axs[1].collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)  # adjust tick font size
    cbar.set_label("Δ Precision", fontsize=14, fontweight='bold')  # label font siz


    plt.savefig("acc_vs_subjects.pdf", dpi=300, bbox_inches="tight")
    return



if __name__ == "__main__":
    # acc_vs_steps()
    
    # acc_vs_temp())
    
    # solving_probs_temp()
    
    subject_vs_acc()

    # solving_probs_matched()
