## GRPO Math Reasoning Evaluation and Analysis

This folder contains analysis scripts for evaluating mathematical reasoning capabilities of LLMs trained with GRPO. The codebase is organized into three main folders:

## 📁 Folder Structure

### `analysis/` - Error Analysis and Human Annotation

This folder contains analysis scripts and data needed to reproduce the plots that analyze model failure modes and the effectiveness of GRPO training.

#### Key Scripts:

**`human_annotation.py`** - Plan vs Execution Failure Analysis
- Creates batch requests to GPT models for grading mathematical solutions on two key dimensions:
  1. **Plan/Direction**: Does the solution attempt the correct approach? (YES/NO)
  2. **Execution**: Does the solution correctly execute the required steps without critical math errors? (YES/NO/N.A.)
- Uses OpenAI batch API to efficiently process large numbers of responses
- Generates ground truth solution summaries using GPT-4
- Produces comparative analysis between pre-GRPO and post-GRPO models
- Key functions:
  - `gpt_idea_summary()`: Generates high-level solution approaches using GPT-4
  - `batch_request_json_creation()`: Creates batch grading requests
  - `compare_models()`: Analyzes plan vs execution success rates across model sizes

**`execution_breakdown.py`** - Detailed Error Categorization
- Builds on the human annotation analysis by providing deeper categorization of execution failures
- For responses that failed execution, categorizes mistakes into three types:
  1. **Basic mathematical factual mistakes** (elementary vs high school level)
  2. **Basic logic mistakes** (non-mathematical reasoning errors)
- Uses a detailed rubric to distinguish between computational errors and fundamental conceptual mistakes
- Generates mistake reduction plots showing GRPO's impact on different error types
- Key functions:
  - `batch_request_json_creation()`: Creates detailed rubric-based grading requests
  - `plot_mistake_drops()`: Visualizes mistake reduction across categories

**`plot.py`** - Visualization and Plotting Functions
- Contains various plotting functions that reproduce the coverage-wall and temperature distillation plots from the paper
- Generates publication-ready figures with seaborn styling
- Key functions:
  - `solving_probs_temp()`: Creates temperature vs precision coverage plots
  - `solving_probs_matched()`: Generates matched problem comparison plots
  - `subject_vs_acc()`: Creates subject-wise accuracy heatmaps
  - `compare_models()`: Produces model comparison visualizations

### `eval/` - Model Evaluation Pipeline

This folder contains the complete evaluation pipeline for generating model responses and computing performance metrics.

#### Key Scripts:

**`generate_responses.py`** - Response Generation
- Uses VLLM for efficient model inference with support for various sampling strategies
- Outputs: JSON files containing problems, solutions, and model responses
- Command line interface for easy experimentation:
```bash
python generate_responses.py --model_name qwen-2.5-1.5b-instruct \
                            --dataset_name math_500 \
                            --temperature 1.0 \
                            --n 64
```

**`extract_solving_probs.py`** - Performance Metrics Computation
- Computes solving probabilities and pass@k metrics from model responses
- Implements unbiased pass@k estimation using combinatorial methods
- Key functions:
  - `estimate_pass_at_k()`: Unbiased estimator for pass@k on single problems
  - `compute_pass_at_k_batch()`: Batch computation across multiple problems
  - `compute_solving_probs()`: Extracts precision rates from response correctness
  - `plot_solving_probs()`: Creates delta plots showing GRPO improvements

**`plot_temp_covg.py`** - Coverage-Wall Analysis
- Creates the main (three-panel) coverage-wall (Pre-GRPO coverage, Post-GRPO coverage, and Pass@K comparison) and temperature distillation plots
- Shows how model performance varies with temperature and sampling strategies

**`utils.py`** - Utility Functions and Configurations
- Contains essential utility functions for the evaluation pipeline:
  - `get_model_path()`: Maps model names to HuggingFace paths or local checkpoints
  - `get_dataset()`: Loads and preprocesses mathematical reasoning datasets
  - `get_prompt_format()`: Handles different prompting strategies
  - Model path mappings for 50+ model variants including:
    - Base models (Qwen-2.5, LLaMA-3.1, DeepSeek, etc.)
    - GRPO-trained checkpoints at different training steps
    - Math-specialized models
- Mathematical answer verification and extraction functions

#### Data Organization:
- `data/`: Contains generated responses organized by model, dataset, and sampling configuration
  - Structure: `{model_name}/{dataset_name}/temp={temperature}_seed={seed}/`
  - Each directory contains `data.json` with responses and `config.yaml` with generation parameters

### `verl_scripts/` - GRPO Training Configurations

Contains VERL config files used for GRPO training, provided for reproducibility.

## 🚀 Quick Start

### 1. Generate Model Responses
```bash
cd eval/
python generate_responses.py --model_name qwen-2.5-1.5b-instruct \
                            --dataset_name math_500 \
                            --temperature 1.0 \
                            --n 64 \
                            --seed 0
```

### 2. Extract Performance Metrics
```bash
python extract_solving_probs.py  # Modify script to point to your data files
```

### 3. Create Analysis Plots
```bash
cd analysis/
python plot.py  # Generates various performance comparison plots
```

### 4. Run Human Annotation Analysis
```bash
python human_annotation.py --model_size 1.5b --result_file pre_temp=1.0_n=64.json
```


