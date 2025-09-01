# Prompt Auto-tuning CLI Guide

This guide explains how to execute prompt auto-tuning from the command line.

## Key Files

- `run_prompt_tuning.py`: Main CLI script
- `run_background.sh`: Bash script for background execution  
- `run_batch_experiments.py`: Script to run multiple experiments in batch

## Basic Usage

### 1. Single Experiment Execution

```bash
# Run BBH dataset with default settings
python run_prompt_tuning.py --dataset bbh --total_samples 20 --iteration_samples 5 --iterations 10

# Use various options
python run_prompt_tuning.py \
    --dataset mmlu \
    --total_samples 50 \
    --iteration_samples 5 \
    --iterations 10 \
    --model solar \
    --evaluator solar \
    --meta_model solar \
    --output_dir ./results/mmlu_test \
    --use_meta_prompt \
    --evaluation_threshold 0.8
```

### 2. Background Execution

```bash
# Default settings (BBH, 20 samples)
./run_background.sh

# Custom settings
./run_background.sh gsm8k 100

# Real-time log monitoring
tail -f ./results/background_bbh_YYYYMMDD_HHMMSS.log

# Check process status
ps -p $(cat ./results/process_bbh_YYYYMMDD_HHMMSS.pid)
```

### 3. Batch Experiment Execution

```bash
# Create default configuration file
python run_batch_experiments.py --create_config

# Check configuration (without actual execution)
python run_batch_experiments.py --dry_run

# Execute batch experiments
python run_batch_experiments.py --config experiments_config.json
```

## Parameter Description

### Required Parameters
- `--dataset`: Dataset to use (bbh, mmlu, gsm8k, cnn, mbpp, xsum, etc.)

### Sampling Settings
- `--total_samples`: Number of samples to sample from entire data (5, 20, 50, 100, 200)
- `--iteration_samples`: Number of samples to use per iteration (default: 5)
- `--iterations`: Number of iterations (default: 10)

### Model Settings
- `--model`: Main model (solar, gpt4o, claude, local1, local2, solar_strawberry)
- `--evaluator`: Evaluation model (default: solar)
- `--meta_model`: Meta prompt generation model (default: solar)

### Tuning Settings
- `--use_meta_prompt`: Use meta prompt (default: True)
- `--evaluation_threshold`: Evaluation score threshold (default: 0.8)
- `--score_threshold`: Average score threshold (default: None)

### Output Settings
- `--output_dir`: Result storage directory (default: ./results)
- `--seed`: Random seed (default: 42)

## Supported Datasets

| Dataset | Description | Sample Type |
|---------|-------------|-------------|
| `bbh` | Big-Bench Hard | Reasoning problems |
| `mmlu` | Massive Multitask Language Understanding | Multiple choice |
| `mmlu_pro` | MMLU Pro | Advanced multiple choice |
| `gsm8k` | Grade School Math 8K | Math problems |
| `cnn` | CNN/DailyMail | Summarization |
| `mbpp` | Mostly Basic Python Programming | Coding |
| `xsum` | Extreme Summarization | Summarization |
| `truthfulqa` | TruthfulQA | Truthfulness evaluation |
| `hellaswag` | HellaSwag | Common sense reasoning |
| `humaneval` | HumanEval | Coding evaluation |
| `samsum` | Samsung Summary | Conversation summarization |
| `meetingbank` | MeetingBank | Meeting summarization |

## Execution Examples

### Example 1: Small-scale Test
```bash
# Quick test with BBH dataset
python run_prompt_tuning.py \
    --dataset bbh \
    --total_samples 5 \
    --iteration_samples 3 \
    --iterations 3 \
    --output_dir ./results/quick_test
```

### Example 2: Medium-scale Experiment
```bash
# Standard experiment with MMLU dataset
python run_prompt_tuning.py \
    --dataset mmlu \
    --total_samples 50 \
    --iteration_samples 5 \
    --iterations 10 \
    --model solar \
    --evaluator solar \
    --meta_model solar \
    --output_dir ./results/mmlu_standard
```

### Example 3: Large-scale Experiment (Background)
```bash
# Large-scale experiment with GSM8K dataset
nohup python run_prompt_tuning.py \
    --dataset gsm8k \
    --total_samples 200 \
    --iteration_samples 10 \
    --iterations 20 \
    --output_dir ./results/gsm8k_large \
    > gsm8k_large.log 2>&1 &
```

## Result Files

After execution, the following files are generated:

- `results_DATASET_TIMESTAMP.csv`: Complete result data
- `cost_summary_DATASET_TIMESTAMP.csv`: Cost summary
- `best_prompt_DATASET_TIMESTAMP.json`: Best performance prompt
- `config_DATASET_TIMESTAMP.json`: Experiment configuration
- `prompt_tuning_TIMESTAMP.log`: Execution log

## Batch Experiment Configuration Example

`experiments_config.json` file example:

```json
{
  "experiments": [
    {
      "name": "BBH_Small_Test",
      "dataset": "bbh",
      "total_samples": 20,
      "iteration_samples": 5,
      "iterations": 10,
      "model": "solar",
      "evaluator": "solar",
      "meta_model": "solar",
      "output_dir": "./results/bbh_small",
      "enabled": true
    },
    {
      "name": "MMLU_Medium_Test",
      "dataset": "mmlu",
      "total_samples": 50,
      "iteration_samples": 5,
      "iterations": 10,
      "model": "solar",
      "evaluator": "solar",
      "meta_model": "solar",
      "output_dir": "./results/mmlu_medium",
      "enabled": true
    }
  ],
  "global_settings": {
    "use_meta_prompt": true,
    "evaluation_threshold": 0.8,
    "score_threshold": null,
    "seed": 42,
    "delay_between_experiments": 60
  }
}
```

## Precautions

1. **API Key Setup**: API keys for the models you use must be configured in the `.env` file.
   - `SOLAR_API_KEY`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `SOLAR_STRAWBERRY_API_KEY`

2. **Memory Usage**: Sufficient memory is required when using large datasets or many samples.

3. **Execution Time**: Execution time varies greatly depending on the number of iterations and samples.

4. **Cost Management**: When using API-based models, carefully monitor token usage and costs.

## Troubleshooting

### Common Errors
- **ModuleNotFoundError**: Install dependencies with `pip install -r requirements.txt`
- **API Key Error**: Check API keys in `.env` file
- **Insufficient Memory**: Reduce sample count or use smaller datasets
- **Permission Error**: Grant execution permission with `chmod +x run_background.sh`

### Log Checking
```bash
# Real-time log monitoring
tail -f ./results/*.log

# Check only error logs
grep -i error ./results/*.log
``` 