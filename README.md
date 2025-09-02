<div align="center">
  <img src="img/title.jpg" alt="ZERA: Zero-prompt Evolving Refinement Agent" width="800px">
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-EMNLP%202025-red.svg)](TBD)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-orange.svg)](TBD)
[![Stars](https://img.shields.io/github/stars/younatics/zera-agent?style=social)](https://github.com/younatics/zera-agent)
[![Forks](https://img.shields.io/github/forks/younatics/zera-agent?style=social)](https://github.com/younatics/zera-agent)

**🎯 The First Joint System-User Prompt Optimization Agent**  
**🚀 From Zero Instructions to Structured Prompts via Self-Refining Optimization**

[![Try it online](https://img.shields.io/badge/Try%20it%20online-Streamlit-red.svg)](TBD)

</div>

# ZERA: Zero-prompt Evolving Refinement Agent

## 🎯 Overview

<div align="center">

**ZERA** is the **first-of-its-kind** prompt auto-tuning agent that revolutionizes how we approach prompt engineering. Unlike traditional methods that require extensive manual crafting, ZERA starts from **zero instructions** and automatically evolves into **high-performance, structured prompts** through intelligent self-refinement.

</div>

### ✨ **What Makes ZERA Special?**

- 🚀 **Zero to Hero**: Start with minimal instructions, end with expert-level prompts
- 🔄 **Self-Evolving**: Continuously improves prompts through automated critique and refinement
- 🎯 **Joint Optimization**: Simultaneously optimizes both system and user prompts
- ⚡ **Lightning Fast**: Achieves high-quality results with only 5-20 samples
- 🧠 **Principle-Based**: Uses 8 evaluation principles for consistent quality
- 📊 **Weighted Scoring**: Adaptive importance weighting for each principle
- 🌟 **Model Agnostic**: Works with any LLM (GPT-4, Claude, Solar, LLaMA, etc.)

### 🎬 **See ZERA in Action**

| Task Type | Before (Zero Prompt) | After (ZERA Optimized) |
|-----------|---------------------|------------------------|
| **Math Reasoning** | "Solve this" | "You are an expert mathematician. Analyze the problem step-by-step, show your work clearly, and provide a comprehensive solution with explanations." |
| **Code Generation** | "Write code" | "You are a senior software engineer. Write clean, efficient, and well-documented code. Include error handling, edge cases, and follow best practices." |
| **Text Summarization** | "Summarize this" | "You are a professional editor. Create concise, accurate summaries that capture key points while maintaining readability and coherence." |

---

## 📚 Research Paper

<div align="center">

🎉 **Congratulations! ZERA has been accepted to EMNLP 2025 Main Conference!** 🎉

</div>

### 📖 **Paper Details**

**Title**: ZERA: Zero-prompt Evolving Refinement Agent – From Zero Instructions to Structured Prompts via Principle-based Optimization  
**Conference**: **EMNLP 2025 Main Conference (Main Track)**  
**Status**: ✅ **Accepted**  
**Authors**: Seungyoun Yi, Minsoo Khang, Sungrae Park

<div align="center">

[![Paper Badge](https://img.shields.io/badge/EMNLP%202025-Main%20Track-blue?style=for-the-badge&logo=academia)](TBD)
[![arXiv Badge](https://img.shields.io/badge/arXiv-coming%20soon-orange?style=for-the-badge)](TBD)

</div>

### Research Contribution
- **Joint Optimization**: Unlike prior APO (Automatic Prompt Optimization) methods that only refine user prompts, ZERA jointly optimizes both **system and user prompts**.
- **Principle-based Evaluation**: Introduces eight general evaluation principles (Correctness, Reasoning Quality, Conciseness, etc.) with adaptive weighting to guide prompt refinement.
- **Self-Refining Framework**: Iterative loop of **PCG (Principle-based Critique Generation)** and **MPR (Meta-cognitive Prompt Refinement)** enables evolution from minimal “zero” prompts to structured, task-optimized prompts.
- **Efficiency**: Achieves high-quality prompts with only **5–20 samples** and short iteration cycles.

### 🏆 **Performance Results**

ZERA has been extensively benchmarked and consistently outperforms state-of-the-art methods:

#### 📊 **Model Coverage**
- **5 LLMs**: GPT-3.5, GPT-4o, LLaMA-3.1, Qwen-2.5, Mistral-7B
- **9 Datasets**: MMLU, GSM8K, BBH, CNN/DailyMail, SAMSum, MBPP, HumanEval, TruthfulQA, HellaSwag

#### 🚀 **Key Advantages**
- **Consistent Performance**: Outperforms recent APO methods across all task types
- **Rapid Convergence**: Achieves optimal results with minimal samples
- **Strong Generalization**: Works across diverse domains without task-specific tuning
- **Zero-Shot Capability**: Starts from minimal instructions, no handcrafted prompts needed

📎 [Read the Full Paper (EMNLP 2025)](TBD)

---


## 🔄 Core Concept: Self-Refining Optimization

<div align="center">

ZERA implements a revolutionary **Self-Refining Optimization** process that transforms minimal instructions into expert-level prompts through intelligent iteration.

</div>

### 🎯 **The ZERA Loop**

<div align="center">

![ZERA Concept](img/ZERA_Concept_v2.png)

*ZERA's iterative prompt refinement process: PCG (Principle-based Critique Generation) → MPR (Meta-cognitive Prompt Refinement) → Enhanced Prompt*

</div>

### 🔧 **How It Works**

1. **🔄 PCG (Principle-based Critique Generation)**
   - Evaluates prompt performance against 8 evaluation principles
   - Generates detailed critiques with scores, analysis, and suggestions
   - Provides weighted feedback based on principle importance

2. **⚡ MPR (Meta-cognitive Prompt Refinement)**
   - Uses critiques to intelligently refine both system and user prompts
   - Leverages historical best prompts and prompt replay data
   - Maintains consistency and improves prompt quality iteratively

3. **♾️ Continuous Refinement Loop**
   - Task samples → Inference → Evaluation → Critique → Refinement
   - Each iteration produces better prompts based on principle-based feedback
   - Rapid convergence to optimal prompts with minimal samples

### 📋 **8 Evaluation Principles**

ZERA evaluates prompts using eight comprehensive principles, each with adaptive weighting:

| Principle | Weight | Description | Focus Area |
|-----------|--------|-------------|------------|
| **Meaning** | 0.15 | Captures key details and core information | Content accuracy |
| **Completeness** | 0.10 | Covers all essential aspects comprehensively | Information coverage |
| **Expression** | 0.05 | Uses appropriate tone and style | Communication quality |
| **Faithfulness** | 0.20 | Stays true to source without fabrication | Source adherence |
| **Conciseness** | 0.20 | Maintains brevity while being complete | Efficiency |
| **Correctness** | 0.10 | Provides accurate and factual information | Factual accuracy |
| **Structural** | 0.05 | Organizes content in logical structure | Organization |
| **Reasoning** | 0.15 | Demonstrates clear logical thinking | Logical flow |

Each principle contributes to the overall prompt quality score, guiding the refinement process toward optimal performance.

---

## Directory Structure and Roles

```
agent/
  app/           # Streamlit-based web UI and state management
  common/        # Common utilities including API clients
  core/          # Core logic for prompt tuning and iteration result management
  dataset/       # Various benchmark datasets and data loaders
  prompts/       # System/user/meta prompt templates
  test/          # Unit test code
  __init__.py    # Package initialization

evaluation/
  base/                # Common base for evaluation system and execution scripts
  dataset_evaluator/   # Dataset-specific evaluators (LLM-based)
    bert/              # BERTScore-based prompt comparison
    llm_judge/         # LLM Judge-based comparison results
  llm_judge/           # LLM Judge evaluation result CSVs
  examples/            # Evaluation and tuning example code
  results/             # Evaluation result storage
  samples/             # Sample data

scripts/               # Command-line interface tools and utilities
  run_prompt_tuning.py      # CLI for prompt tuning experiments
  run_batch_experiments.py  # Batch experiment execution
  update_results.py         # Result update utilities
  run_background.sh         # Background process management
```

### agent directory
- **app/**: Streamlit-based web interface and state management
- **common/**: Common client for communicating with various LLM APIs
- **core/**: Core logic for prompt auto-tuning and iteration result management
- **dataset/**: Various benchmark dataset loaders and data folders
- **prompts/**: System/user/meta/evaluation prompt templates
- **test/**: Prompt tuner test code

### evaluation directory
- **base/**: Common base classes for evaluation system (`BaseEvaluator`) and execution scripts (`main.py`)
- **dataset_evaluator/**: LLM evaluators for each dataset (e.g., `gsm8k_evaluator.py`, `mmlu_evaluator.py`, etc.)
  - **bert/**: Prompt comparison using BERTScore and results (`bert_compare_prompts.py`, `zera_score.json`, `base_score.json`, etc.)
  - **llm_judge/**: LLM Judge-based comparison result storage
- **llm_judge/**: Comparison result CSVs generated by LLM Judge
- **examples/**: Dataset-specific evaluation/tuning example code and execution methods
- **results/**: Evaluation result storage folder
- **samples/**: Sample data

---

## Key Features

- **Prompt Auto-tuning**:  
  - Iteratively improve system/user prompts to maximize LLM performance
  - Utilize meta-prompts to guide LLMs to directly improve prompts themselves

- **Support for Various Models and Datasets**:  
  - Support for various models including OpenAI GPT, Anthropic Claude, Upstage Solar, local LLMs
  - Built-in benchmark datasets including MMLU, GSM8K, CNN, MBPP, TruthfulQA

- **Automated Output Evaluation**:  
  - Automatically evaluate LLM outputs using 8 evaluation criteria (accuracy, completeness, expression, reliability, conciseness, correctness, structural consistency, reasoning quality)
  - Improve prompts based on evaluation results

- **Various Evaluation Methods**:
  - **LLM-based Evaluation**: LLMs directly perform correctness assessment, scoring, and detailed evaluation for each dataset
  - **BERTScore-based Evaluation**: Compare output similarity (F1, Precision, Recall, etc.) between prompts using BERT embeddings
  - **LLM Judge-based Evaluation**: LLMs directly compare outputs from two prompts to determine winner/loser and reasons

- **Web UI**:  
  - Intuitive experiment management and result visualization based on Streamlit

---

## Evaluation System Usage

### 1. LLM Evaluation Execution

You can execute LLM evaluation with various datasets and prompts through `evaluation/base/main.py`.

```bash
python evaluation/base/main.py --dataset <dataset_name> --model <model_name> --model_version <version> \
  --base_system_prompt <existing_system_prompt> --base_user_prompt <existing_user_prompt> \
  --zera_system_prompt <zera_system_prompt> --zera_user_prompt <zera_user_prompt> \
  --num_samples <sample_count>
```

- Evaluation results are stored in `evaluation/results/`.
- You can compare prompt performance using various metrics like accuracy, ROUGE, etc.

### 2. BERTScore-based Prompt Comparison

Running `evaluation/dataset_evaluator/bert/bert_compare_prompts.py` allows you to compare ZERA prompt and existing prompt outputs using BERTScore.

```bash
python evaluation/dataset_evaluator/bert/bert_compare_prompts.py
```

- Results are saved as `comparison_results.csv`.

### 3. LLM Judge-based Comparison

You can check results where LLMs directly compare outputs from two prompts (winner, reasons, etc.) in `evaluation/llm_judge/comparison_results.csv`.

### 4. Example Execution

The `evaluation/examples/` directory contains example code for each dataset.

```bash
python evaluation/examples/<dataset>_example.py
```

- Requires `requirements.txt` installation and `.env` environment variable setup before running examples

---

## 🚀 Quick Start

<div align="center">

**Get up and running with ZERA in under 5 minutes!** ⚡

</div>



### 1. Clone and Setup
```bash
git clone https://github.com/younatics/zera-agent.git
cd zera-agent
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root with your API keys:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
SOLAR_API_KEY=your_solar_api_key_here
SOLAR_STRAWBERRY_API_KEY=your_solar_strawberry_api_key_here

# Optional: Local model configuration
LOCAL_MODEL_ENDPOINT=http://localhost:8000/v1
LOCAL_MODEL_API_KEY=your_local_api_key_here

# Optional: Slack notifications
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
SLACK_CHANNEL=#experiments
```

**Note**: You only need to set the API keys for the models you plan to use.

### 3. Run Your First Experiment
```bash
# Quick test with BBH dataset
python scripts/run_prompt_tuning.py \
  --dataset bbh \
  --total_samples 10 \
  --iterations 3 \
  --model solar
```

### 4. Explore with Web UI
```bash
streamlit run agent/app/streamlit_app.py
```

## Installation and Execution

1. Install dependencies
   ```
   pip install -r requirements.txt
   ```

2. Set environment variables  
   Enter OpenAI, Anthropic, etc. API keys in `.env` file

3. Run web UI
   ```
   streamlit run agent/app/streamlit_app.py
   ```

4. Run CLI tools (optional)
   ```bash
   # Run prompt tuning experiment
   python scripts/run_prompt_tuning.py --dataset bbh --total_samples 20 --iterations 5 --model solar
   
   # Run batch experiments
   python scripts/run_batch_experiments.py --config experiments_config.json
   
   # Update results
   python scripts/update_results.py
   ```

---

## Usage Examples

- Automatically generate optimal prompts for new tasks
- Automate LLM benchmark experiments and result comparison
- Prompt engineering research and experiments
- Quantitative/qualitative prompt performance comparison using various evaluation methods (LLM, BERT, LLM Judge)

---

## Troubleshooting

### Common Issues

#### 🔑 **API Key Errors**
```bash
Error: No API key found for model 'solar'
```
**Solution**: Ensure your `.env` file contains the correct API key for the model you're using.

#### 📦 **Import Errors**
```bash
ModuleNotFoundError: No module named 'agent'
```
**Solution**: Make sure you're running commands from the project root directory, not from subdirectories.

#### 💾 **Memory Issues**
```bash
MemoryError: Unable to allocate array
```
**Solution**: Reduce the `--total_samples` or `--iteration_samples` parameters.

#### ⏱️ **Timeout Errors**
```bash
RequestTimeout: Request timed out
```
**Solution**: Check your internet connection and API rate limits.

#### 📊 **Evaluation Errors**
```bash
EvaluationError: Failed to evaluate response
```
**Solution**: Verify your evaluation prompts are properly formatted and the model can access them.

### Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/younatics/zera-agent/issues)
- **Discussions**: [Join community discussions](https://github.com/younatics/zera-agent/discussions)
- **Documentation**: Check the [scripts/README.md](scripts/README.md) for CLI usage details

## 🤝 Community & Contributing

<div align="center">

**Join the ZERA community and help shape the future of prompt engineering!** 🌟

</div>

### 🚀 **Get Involved**

- 🐛 **Report Bugs**: [GitHub Issues](https://github.com/younatics/zera-agent/issues)
- 💡 **Request Features**: [Feature Requests](https://github.com/younatics/zera-agent/discussions)
- 📚 **Ask Questions**: [Q&A Discussions](https://github.com/younatics/zera-agent/discussions)
- 🔧 **Contribute Code**: [Pull Requests](https://github.com/younatics/zera-agent/pulls)
- 📖 **Improve Docs**: [Documentation PRs](https://github.com/younatics/zera-agent/pulls)



### 📧 **Stay Connected**

- 📧 **Email**: [younatics@gmail.com](mailto:younatics@gmail.com)
- 🐦 **Twitter**: [@ZERAAgent](https://twitter.com/ZERAAgent)

### 🏆 **Contributors**

We welcome contributions from the community! See our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved.

## Citation

If you use ZERA in your research, please cite our paper:

```bibtex
@inproceedings{yi2025zera,
  title={ZERA: Zero-prompt Evolving Refinement Agent – From Zero Instructions to Structured Prompts via Principle-based Optimization},
  author={Yi, Seungyoun and Khang, Minsoo and Park, Sungrae},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  publisher={Association for Computational Linguistics}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 **Ready to Transform Your Prompt Engineering?**

<div align="center">

**ZERA is not just another tool—it's a revolution in how we approach AI prompting.** 🚀

</div>

### 🚀 **What's Next?**

1. **🎯 Try ZERA**: Run your first experiment in minutes
2. **📚 Read the Paper**: Dive deep into the research
3. **🌟 Star the Repo**: Show your support
4. **🤝 Contribute**: Help shape the future of prompt engineering
5. **📢 Share**: Let others know about ZERA

### 🔮 **The Future of Prompt Engineering**

With ZERA, the era of manual prompt crafting is over. Welcome to the future where:
- **Zero instructions** become **expert-level prompts**
- **Manual tuning** becomes **automated optimization**
- **Trial and error** becomes **intelligent refinement**
- **Domain expertise** becomes **universal capability**

---

<div align="center">

**ZERA: Zero-prompt Evolving Refinement Agent**  
*From Zero Instructions to Structured Prompts via Self-Refining Optimization*

---

*Ready to experience the future of prompt engineering?* 🚀

</div> 