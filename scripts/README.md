<div align="center">
  <img src="../img/title.jpg" alt="ZERA: Zero-prompt Evolving Refinement Agent" width="600px">
</div>

# Scripts Directory

This directory contains command-line interface tools and utility scripts for the zera-agent project.

## Available Scripts

### 🚀 `run_prompt_tuning.py`
Command-line interface for running prompt tuning experiments.

**Usage:**
```bash
python scripts/run_prompt_tuning.py --dataset bbh --total_samples 20 --iteration_samples 5 --iterations 10 --model solar --evaluator solar --meta_model solar --output_dir ./results
```

**Options:**
- `--dataset`: Dataset to use (bbh, mmlu, mmlu_pro, cnn, gsm8k, mbpp, xsum, truthfulqa, hellaswag, humaneval, samsum, meetingbank)
- `--total_samples`: Total number of samples to use
- `--iteration_samples`: Number of samples per iteration
- `--iterations`: Number of tuning iterations
- `--model`: Model to use for tuning
- `--evaluator`: Model to use for evaluation
- `--meta_model`: Model to use for meta prompt generation
- `--output_dir`: Directory to save results

### 🔄 `run_batch_experiments.py`
Script to run multiple prompt tuning experiments in batch.

**Usage:**
```bash
python scripts/run_batch_experiments.py --config experiments_config.json
```

**Features:**
- Batch execution of multiple experiments
- Configurable experiment parameters
- Progress tracking and logging
- Error handling and recovery

### 📊 `update_results.py`
Utility script to update evaluation results.

**Usage:**
```bash
python scripts/update_results.py
```

**Purpose:**
- Re-evaluate existing results with updated models
- Update correctness scores
- Generate new result files

### ⏰ `run_background.sh`
Shell script for running experiments in the background.

**Usage:**
```bash
bash scripts/run_background.sh
```

**Features:**
- Background process management
- Log file generation
- Process monitoring

## Running Scripts

All scripts should be run from the project root directory:

```bash
cd /path/to/zera-agent
python scripts/script_name.py [options]
```

## Dependencies

Make sure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

## Environment Variables

Set up your environment variables in a `.env` file:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Examples

### Quick Start
```bash
# Run a simple BBH experiment
python scripts/run_prompt_tuning.py --dataset bbh --total_samples 10 --iterations 3 --model solar

# Run batch experiments
python scripts/run_batch_experiments.py --config experiments_config.json
```

### Advanced Usage
```bash
# Custom model configuration
python scripts/run_prompt_tuning.py \
  --dataset mmlu \
  --total_samples 50 \
  --iterations 5 \
  --model gpt4o \
  --evaluator claude \
  --meta_model solar \
  --output_dir ./custom_results
```

### Real-World Scenarios

#### 🧮 **Math Problem Solving (GSM8K)**
```bash
python scripts/run_prompt_tuning.py \
  --dataset gsm8k \
  --total_samples 100 \
  --iterations 10 \
  --model gpt4o \
  --evaluator gpt4o \
  --meta_model gpt4o \
  --evaluation_threshold 0.9 \
  --output_dir ./results/gsm8k_math_expert
```

#### 📚 **Multiple Choice Questions (MMLU)**
```bash
python scripts/run_prompt_tuning.py \
  --dataset mmlu \
  --total_samples 200 \
  --iterations 15 \
  --model claude \
  --evaluator claude \
  --meta_model claude \
  --evaluation_threshold 0.85 \
  --output_dir ./results/mmlu_knowledge_expert
```

#### 💻 **Programming Problems (MBPP)**
```bash
python scripts/run_prompt_tuning.py \
  --dataset mbpp \
  --total_samples 50 \
  --iterations 8 \
  --model solar \
  --evaluator solar \
  --meta_model solar \
  --evaluation_threshold 0.8 \
  --output_dir ./results/mbpp_code_expert
```

#### 📰 **News Summarization (CNN)**
```bash
python scripts/run_prompt_tuning.py \
  --dataset cnn \
  --total_samples 80 \
  --iterations 12 \
  --model gpt4o \
  --evaluator claude \
  --meta_model solar \
  --evaluation_threshold 0.75 \
  --output_dir ./results/cnn_summary_expert
```

### Cost Optimization Examples

#### 💰 **Budget-Conscious Experiment**
```bash
# Use smaller sample sizes to reduce costs
python scripts/run_prompt_tuning.py \
  --dataset bbh \
  --total_samples 20 \
  --iterations 5 \
  --model solar \
  --evaluation_threshold 0.7 \
  --output_dir ./results/bbh_budget_test
```

#### 🚀 **High-Performance Experiment**
```bash
# Use larger samples for better results (higher cost)
python scripts/run_prompt_tuning.py \
  --dataset mmlu \
  --total_samples 500 \
  --iterations 20 \
  --model gpt4o \
  --evaluation_threshold 0.95 \
  --output_dir ./results/mmlu_high_performance
```
