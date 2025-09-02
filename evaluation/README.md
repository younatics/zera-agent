# Evaluation Directory

This directory contains the comprehensive evaluation system for the Zera Agent, providing multiple evaluation methodologies and result analysis tools.

## Directory Structure

```
evaluation/
├── base/                    # Common base classes and execution scripts
├── dataset_evaluator/       # Dataset-specific evaluators
│   ├── bert/               # BERTScore-based evaluation
│   └── llm_judge/          # LLM Judge-based evaluation
├── examples/                # Usage examples for each dataset
├── llm_judge/              # LLM Judge evaluation results
├── code_analysis/          # Code analysis and evaluation tools
├── number_extraction/       # Number extraction utilities
├── results/                 # Evaluation result storage
├── samples/                 # Sample data for testing
└── main.py                  # Main evaluation execution script
```

## Core Components

### 🏗️ **base/** - Foundation Classes
- **`evaluator.py`**: Base evaluator class with common evaluation logic
- **`dataset_loader.py`**: Base dataset loading functionality
- **`main.py`**: Main execution script for evaluation experiments

### 🔍 **dataset_evaluator/** - Specialized Evaluators
- **`bbh_evaluator.py`**: Big-Bench Hard task evaluator
- **`gsm8k_evaluator.py`**: Math problem evaluator
- **`mmlu_evaluator.py`**: Multiple choice evaluator
- **`cnn_dailymail_evaluator.py`**: Summarization evaluator
- **`mbpp_evaluator.py`**: Programming problem evaluator
- **`truthfulqa_evaluator.py`**: Truthfulness evaluator
- **`hellaswag_evaluator.py`**: Common sense reasoning evaluator
- **`humaneval_evaluator.py`**: Code evaluation evaluator
- **`samsum_evaluator.py`**: Conversation summarization evaluator
- **`xsum_evaluator.py`**: Extreme summarization evaluator

#### **bert/** - BERTScore Evaluation
- **`bert_compare_prompts.py`**: Compare prompts using BERTScore
- **`zera_score.json`**: Zera prompt BERTScore results
- **`base_score.json`**: Base prompt BERTScore results
- **`comparison_results.csv`**: Comparative analysis results

#### **llm_judge/** - LLM Judge Evaluation
- **`llm_judge_evaluator.py`**: LLM-based prompt comparison
- **`comparison_results.csv`**: LLM Judge evaluation results

### 📚 **examples/** - Usage Examples
- **`gsm8k/`**: Math problem solving examples
- **`mmlu/`**: Multiple choice problem examples
- **`bbh/`**: Big-Bench Hard task examples
- **`cnn_dailymail/`**: News summarization examples
- **`mbpp/`**: Programming problem examples
- **`README.md`**: Examples documentation

### 📊 **llm_judge/** - Evaluation Results
- **`comparison_results.csv`**: LLM Judge comparison results
- Winner/loser analysis between prompts
- Detailed reasoning and explanations

### 🔬 **code_analysis/** - Code Evaluation Tools
- **`analyze_hellaswag_json_eval.py`**: HellaSwag JSON evaluation analysis
- **`analyze_humaneval_json_eval.py`**: HumanEval JSON evaluation analysis
- **`analyze_mbpp_json_eval.py`**: MBPP JSON evaluation analysis

### 📈 **number_extraction/** - Result Analysis
- **`*.csv`**: Extracted numerical results for each dataset
- **`summarize_prompt_tuning_results.py`**: Result summarization tool

### 💾 **results/** - Result Storage
- Evaluation result files in JSON format
- Organized by dataset, model, and timestamp
- Structured for easy analysis and comparison

### 🧪 **samples/** - Test Data
- Sample datasets for testing and development
- Small-scale examples for quick validation

## Evaluation Methodologies

### 1. **LLM-based Evaluation**
- Direct correctness assessment by LLMs
- Detailed scoring across multiple criteria
- Context-aware evaluation for each dataset type

**Usage:**
```bash
python evaluation/base/main.py \
  --dataset bbh \
  --model gpt4o \
  --base_system_prompt "You are a helpful assistant..." \
  --base_user_prompt "Solve this problem..." \
  --zera_system_prompt "You are an expert problem solver..." \
  --zera_user_prompt "Analyze and solve this problem..." \
  --num_samples 20
```

### 2. **BERTScore-based Evaluation**
- Semantic similarity comparison using BERT embeddings
- F1, Precision, and Recall metrics
- Quantitative prompt performance analysis

**Usage:**
```bash
python evaluation/dataset_evaluator/bert/bert_compare_prompts.py
```

### 3. **LLM Judge-based Evaluation**
- Direct comparison between two prompts
- Winner/loser determination with reasoning
- Qualitative prompt performance analysis

**Features:**
- Head-to-head prompt comparison
- Detailed reasoning for decisions
- Confidence scoring

## Key Features

### 🎯 **Multi-Criteria Evaluation**
- **Accuracy**: Correctness of responses
- **Completeness**: Coverage of required information
- **Expression**: Clarity and coherence
- **Reliability**: Consistency and dependability
- **Conciseness**: Brevity without loss of information
- **Correctness**: Factual accuracy
- **Structural Alignment**: Logical organization
- **Reasoning Quality**: Soundness of logical steps

### 📊 **Comprehensive Metrics**
- **ROUGE Scores**: For summarization tasks
- **Exact Match**: For factual questions
- **Semantic Similarity**: Using BERTScore
- **Custom Metrics**: Dataset-specific evaluation criteria

### 🔄 **Batch Processing**
- Support for multiple experiments
- Automated result collection
- Comparative analysis tools

## Usage Examples

### Quick Evaluation
```bash
# Evaluate a single prompt on BBH dataset
python evaluation/base/main.py \
  --dataset bbh \
  --model solar \
  --base_system_prompt "You are a helpful assistant." \
  --base_user_prompt "Solve this reasoning problem." \
  --num_samples 10
```

### Comparative Analysis
```bash
# Compare two prompts using LLM Judge
python evaluation/dataset_evaluator/llm_judge/llm_judge_evaluator.py \
  --dataset gsm8k \
  --model gpt4o \
  --prompt1 "You are a math tutor." \
  --prompt2 "You are a mathematics expert." \
  --num_samples 20
```

### BERTScore Analysis
```bash
# Analyze prompt similarity using BERTScore
python evaluation/dataset_evaluator/bert/bert_compare_prompts.py \
  --dataset mmlu \
  --base_prompt "Answer this question." \
  --zera_prompt "Analyze and answer this question step by step."
```

## Configuration

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Model Configuration
- **OpenAI Models**: GPT-4, GPT-3.5-turbo
- **Anthropic Models**: Claude-3, Claude-2
- **Upstage Models**: Solar, Solar-Strawberry
- **Local Models**: Custom local LLM endpoints

### Dataset Configuration
Each dataset can be configured with:
- Sample size and selection strategy
- Evaluation criteria and weights
- Output format and storage options

## Result Analysis

### Output Formats
- **JSON**: Detailed evaluation results
- **CSV**: Tabular data for analysis
- **Logs**: Execution logs and debugging information

### Analysis Tools
- **`summarize_prompt_tuning_results.py`**: Aggregate and summarize results
- **Number extraction scripts**: Extract key metrics for comparison
- **Visualization tools**: Plot and analyze performance trends

## Development

### Adding New Evaluators
1. Create a new evaluator class in `dataset_evaluator/`
2. Inherit from `BaseEvaluator`
3. Implement required methods (`evaluate_response()`, etc.)
4. Add to the evaluator registry

### Adding New Datasets
1. Create dataset-specific evaluator
2. Implement evaluation logic
3. Add to examples and documentation
4. Update configuration files

### Running Tests
```bash
# Run evaluation tests
python -m pytest evaluation/tests/

# Run specific evaluator tests
python -m pytest evaluation/dataset_evaluator/test_bbh_evaluator.py
```

## Performance Considerations

- **API Costs**: Monitor token usage and costs
- **Execution Time**: Large datasets may take significant time
- **Memory Usage**: Consider batch processing for large evaluations
- **Rate Limiting**: Respect API rate limits and implement backoff

## Troubleshooting

### Common Issues
- **Import Errors**: Check Python path and dependencies
- **API Errors**: Verify API keys and rate limits
- **Memory Issues**: Reduce batch size or use streaming
- **Performance Issues**: Optimize evaluation criteria and batch processing

### Debug Mode
Enable detailed logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Analysis
```bash
# Check evaluation logs
tail -f evaluation/results/*.log

# Search for errors
grep -i error evaluation/results/*.log
```

## Contributing

- Follow the existing code structure and patterns
- Add comprehensive tests for new evaluators
- Update documentation for new features
- Ensure backward compatibility for existing evaluations
