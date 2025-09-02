# Agent Directory

This directory contains the core components of the Zera Agent prompt auto-tuning system.

## Directory Structure

```
agent/
├── app/                    # Streamlit-based web UI and state management
├── common/                 # Common utilities including API clients
├── core/                   # Core logic for prompt tuning and iteration result management
├── dataset/                # Various benchmark datasets and data loaders
├── prompts/                # System/user/meta prompt templates
├── test/                   # Unit test code
└── scripts/                # Additional utility scripts
```

## Core Components

### 🖥️ **app/** - Web User Interface
- **`streamlit_app.py`**: Main Streamlit application for interactive prompt tuning
- Provides intuitive experiment management and result visualization
- Real-time progress tracking and cost monitoring
- Interactive prompt editing and configuration

### 🔧 **common/** - Shared Utilities
- **`api_client.py`**: Unified client for various LLM APIs (OpenAI, Anthropic, Solar, etc.)
- **`slack_notify.py`**: Slack notification utilities for experiment monitoring
- Common configuration and utility functions

### 🧠 **core/** - Core Logic
- **`prompt_tuner.py`**: Main prompt tuning engine with iterative improvement
- **`iteration_result.py`**: Data structures for managing iteration results
- Cost tracking, statistics, and performance monitoring

### 📊 **dataset/** - Data Management
- **`bbh_dataset.py`**: Big-Bench Hard dataset loader
- **`mmlu_dataset.py`**: Massive Multitask Language Understanding dataset
- **`gsm8k_dataset.py`**: Grade School Math 8K dataset
- **`cnn_dataset.py`**: CNN/DailyMail summarization dataset
- **`mbpp_dataset.py`**: Mostly Basic Python Programming dataset
- **`truthfulqa_dataset.py`**: TruthfulQA dataset
- **`hellaswag_dataset.py`**: HellaSwag common sense reasoning dataset
- **`humaneval_dataset.py`**: HumanEval coding evaluation dataset
- **`samsum_dataset.py`**: Samsung Summary conversation dataset
- **`meetingbank_dataset.py`**: MeetingBank meeting summarization dataset
- **`xsum_dataset.py`**: Extreme Summarization dataset

### 📝 **prompts/** - Prompt Templates
- **`initial_system_prompt.txt`**: Default system prompt for new experiments
- **`initial_user_prompt.txt`**: Default user prompt for new experiments
- **`evaluation_system_prompt.txt`**: System prompt for evaluation models
- **`evaluation_user_prompt.txt`**: User prompt for evaluation models
- **`meta_system_prompt.txt`**: System prompt for meta prompt generation
- **`meta_user_prompt.txt`**: User prompt for meta prompt generation

### 🧪 **test/** - Testing
- **`test_prompt_tuner.py`**: Unit tests for the prompt tuner
- Test coverage for core functionality

## Key Features

### 🚀 **Prompt Auto-tuning Engine**
- Iterative prompt improvement using meta-prompts
- Automatic evaluation and scoring
- Cost and performance tracking
- Configurable improvement thresholds

### 🔄 **Multi-Model Support**
- OpenAI GPT models (GPT-4, GPT-3.5)
- Anthropic Claude models
- Upstage Solar models
- Local LLM support
- Extensible architecture for new models

### 📈 **Performance Monitoring**
- Real-time cost tracking
- Token usage statistics
- Execution time monitoring
- Performance metrics visualization

### 🎯 **Evaluation System**
- 8-criteria evaluation (accuracy, completeness, expression, reliability, conciseness, correctness, structural consistency, reasoning quality)
- Automated scoring and feedback
- Comparative analysis between prompts

## Usage

### Web Interface
```bash
streamlit run agent/app/streamlit_app.py
```

### Programmatic Usage
```python
from agent.core.prompt_tuner import PromptTuner
from agent.dataset.bbh_dataset import BBHDataset

# Initialize tuner
tuner = PromptTuner(
    model_name="solar",
    evaluator_model_name="solar",
    meta_prompt_model_name="solar"
)

# Load dataset
dataset = BBHDataset()
test_cases = dataset.get_category_data("logical_deduction")

# Run tuning
results = tuner.tune_prompt(
    initial_system_prompt="You are a helpful assistant...",
    initial_user_prompt="Solve this logical problem...",
    initial_test_cases=test_cases,
    num_iterations=5,
    num_samples=10
)
```

## Configuration

### Environment Variables
Set up your API keys in a `.env` file:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
SOLAR_API_KEY=your_solar_key
SOLAR_STRAWBERRY_API_KEY=your_solar_strawberry_key
```

### Model Configuration
Each model can be configured with:
- Model name and version
- API endpoints and authentication
- Rate limiting and retry policies
- Cost tracking settings

## Development

### Adding New Datasets
1. Create a new dataset class in `dataset/`
2. Inherit from base dataset class
3. Implement required methods (`get_data()`, `get_split_data()`, etc.)
4. Add to the main dataset registry

### Adding New Models
1. Extend the `Model` class in `common/api_client.py`
2. Implement required methods (`generate()`, `evaluate()`, etc.)
3. Add configuration in the model registry

### Running Tests
```bash
python -m pytest agent/test/
```

## Architecture

The agent follows a modular architecture:
- **Separation of Concerns**: Each component has a specific responsibility
- **Plugin Architecture**: Easy to add new datasets, models, and evaluation methods
- **Configuration-Driven**: Most behavior is configurable without code changes
- **Event-Driven**: Progress callbacks and notifications for monitoring

## Performance Considerations

- **Memory Usage**: Large datasets may require significant memory
- **API Costs**: Monitor token usage and costs for API-based models
- **Execution Time**: Iterative tuning can take significant time
- **Concurrency**: Support for parallel evaluation and batch processing

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **API Errors**: Check API keys and rate limits
- **Memory Issues**: Reduce dataset size or use streaming
- **Performance Issues**: Adjust batch sizes and concurrency settings

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
