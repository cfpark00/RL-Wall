# Decomposing Elements of Problem Solving: What "Math" Does RL Teach?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and experiments for the research paper "Decomposing Elements of Problem Solving: What 'Math' Does RL Teach?" which investigates how reinforcement learning (RL) affects mathematical reasoning capabilities in large language models.

## 🔬 Research Overview

Mathematical reasoning tasks have become prominent benchmarks for assessing LLM reasoning capabilities, especially with RL methods like GRPO showing significant performance gains. However, accuracy metrics alone don't reveal which problem-solving skills have been internalized.

### Key Contributions

1. **Reasoning Decomposition Framework**: We propose decomposing math problem solving into three fundamental capabilities:
   - **Plan**: Mapping questions to sequences of solution steps
   - **Execute**: Correctly performing solution steps  
   - **Verify**: Identifying the correctness of a solution

2. **Empirical Analysis of RL**: We show that GRPO primarily improves execution on known problems through a "temperature distillation" effect, but fails to solve previously unsolved problems, revealing a "coverage wall".

3. **Synthetic Validation**: We construct a minimal synthetic task that replicates our empirical findings and identifies conditions under which RL can overcome the coverage wall.

### Key Findings

- **Temperature Distillation**: GRPO makes correct solutions more likely regardless of sampling temperature, enhancing execution robustness
- **Coverage Wall**: RL fails to help models solve fundamentally new problems due to insufficient planning skills
- **Execution Enhancement**: RL primarily strengthens execution by reducing spurious correlations and basic errors

## 📁 Repository Structure

```
RL-Wall/
├── eval/                    # Evaluation framework and utilities
│   ├── utils.py            # Core evaluation utilities and verifiers
│   ├── generate_responses.py # Response generation script
│   ├── extract_correct.py   # Answer extraction utilities
│   └── scripts/            # Collection of evaluation scripts for different models
├── synthetic/              # Synthetic environment for controlled experiments
│   ├── make_data_synthetic_v5.ipynb  # Synthetic data generation notebook
│   ├── make_models_v5.py   # Synthetic model creation script
│   ├── eval_f.py & eval_t.py # Evaluation scripts for synthetic models
│   ├── configs/            # YAML training configurations (v5_1.yaml, etc.)
│   ├── sft/               # Supervised fine-tuning code
│   │   ├── run_sft_accelerate.py # SFT training script
│   │   └── lm_tools.py    # Language model utilities
│   └── rl/                # Reinforcement learning setup (VERL framework)
├── tree_vis/              # Solution tree visualization tools
│   ├── make_tree_04_14.ipynb # Interactive tree visualization notebook
│   ├── trees/             # Generated solution tree files
│   └── *.html            # Example visualization files
├── math_rl/               # Mathematical RL experiments (minimal content)
└── README.md
```

## 🚀 Getting Started

### Prerequisites

The repository uses several dependencies. You'll need:

```bash
# Core dependencies
pip install torch transformers datasets numpy pandas
pip install vllm accelerate wandb tqdm
pip install sympy pylatexenc

# For RL training (VERL framework is included)
cd synthetic/rl/verl
pip install -e .

# For evaluation with GPT-based verification
pip install openai
```

## 📋 What's Actually Here

### Evaluation Framework (`eval/`)

- **`utils.py`**: Comprehensive utilities with multiple answer verifiers (VERL, SymPy, GPT-based)
- **`generate_responses.py`**: Script for generating model responses with various parameters
- **`extract_correct.py`**: Utilities for extracting and processing answers
- **`scripts/`**: Collection of bash scripts for running evaluations (e.g., `qwen-1.5b-instruct_temps.sh`)

### Synthetic Environment (`synthetic/`)

- **`make_data_synthetic_v5.ipynb`**: Jupyter notebook for creating synthetic datasets
- **`make_models_v5.py`**: Script for synthetic model creation
- **`eval_f.py`** and **`eval_t.py`**: Evaluation scripts for synthetic experiments
- **`configs/`**: YAML configuration files (v5_1.yaml through v5_17.yaml)
- **`sft/run_sft_accelerate.py`**: Training script using Accelerate
- **`rl/verl/`**: Complete VERL framework for RL training

### Tree Visualization (`tree_vis/`)

- **`make_tree_04_14.ipynb`**: Notebook for generating interactive solution trees
- **Various HTML files**: Pre-generated visualization examples
- **`trees.json`**: Solution tree data

## 🧪 Running Experiments

### Basic Evaluation

You can generate responses using the evaluation framework:

```bash
cd eval
python generate_responses.py \
    --model_name qwen-2.5-1.5b-instruct \
    --dataset_name math_500 \
    --exp_dir ./results/test \
    --temperature 0.1 \
    --n 64
```

### Synthetic Experiments

The synthetic environment can be explored through the notebooks:

```bash
cd synthetic
# Open and run the data generation notebook
jupyter notebook make_data_synthetic_v5.ipynb

# Train a synthetic model (requires proper setup)
python sft/run_sft_accelerate.py configs/v5_1.yaml
```

### Solution Tree Visualization

```bash
cd tree_vis
# Open the visualization notebook
jupyter notebook make_tree_04_14.ipynb
```

## 📊 Key Components

### Evaluation Utilities

The `eval/utils.py` file contains:
- Multiple answer verification methods
- Support for various model architectures (Qwen, Llama, DeepSeek, etc.)
- Batch processing capabilities
- Temperature and sampling analysis tools

### Synthetic Environment Design

The synthetic setup models mathematical reasoning as:
- State-action navigation through transition tables
- Built-in spurious correlations for robustness testing
- Configurable complexity and dimensions

### Visualization Tools

- Interactive HTML-based solution tree visualization
- Statistical analysis of model behavior patterns
- Tools for comparing pre/post-RL model performance

## 📈 Research Findings

Based on the code and experiments in this repository:

1. **GRPO improves precision** through temperature distillation but **doesn't increase coverage**
2. **Models plan well** but **struggle with execution** on high school math
3. **RL reduces basic errors** but doesn't teach new mathematical knowledge
4. **Coverage improvements are possible** under specific conditions (less spurious correlation, more RL data)

## ⚠️ Repository Status

This repository contains the research code and experimental setup. Some components may require additional setup or configuration to run fully. The code represents the state used for the research paper and may need adaptation for different environments or use cases.

## 📝 Citation
Coming Soon



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- VERL framework for efficient RL training
- MATH and GSM8K datasets for evaluation
- Qwen model family for base models

---

For questions about the code or experiments, please open a GitHub issue.