# Generative AI Assignment 1: Prompt Engineering

**Student Name:** Weihao He


This project implements manual and automated prompt engineering techniques on the GSM8K dataset to improve LLM performance on math word problems.

## Project Structure

```
.
├── step1.py                        # Step 1: Download data and evaluate base prompt
├── step2.py                        # Step 2: Manual prompt engineering improvements
├── step3.py                        # Step 3: Automated prompt engineering (OPRO-style)
├── gsm8k_sample_300.json           # Sampled 300 questions from GSM8K
├── base_prompt_results.json        # Results from base prompt evaluation
├── improved_prompt_results.json    # Results from improved prompt
├── cot_prompt_results.json         # Results from chain of thought prompt
├── opro_train_iter1.json           # Iteration 1 training results
├── opro_train_iter2.json           # Iteration 2 training results
├── opro_train_iter3.json           # Iteration 3 training results
├── opro_test_results.json          # Final test set results
├── opro_summary.json               # Summary of automated optimization
├── requirements.txt                # Python dependencies
└── Dockerfile                      # Docker configuration
```

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Docker (optional, for containerized execution)

## Installation

### Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Using Docker

1. Build the Docker image:
```bash
docker build -t prompt-engineering:latest .
```

2. Run the container:
```bash
docker run -e OPENAI_API_KEY="your-api-key-here" prompt-engineering:latest
```

## Running the Project

### Step 1: Baseline Evaluation

Evaluates the base prompt on 300 sampled GSM8K questions:

```bash
python step1.py
```

Output files:
- `gsm8k_sample_300.json` - Sampled dataset
- `base_prompt_results.json` - Results with accuracy metrics

### Step 2: Manual Prompt Engineering

Tests two manual optimization techniques:
1. **Improved Prompt** - With better instructions and reasoning guidance
2. **Chain of Thought (CoT)** - Explicit step-by-step reasoning

```bash
python step2.py
```

Output files:
- `improved_prompt_results.json` - Improved prompt results
- `cot_prompt_results.json` - Chain of thought results

### Step 3: Automated Prompt Engineering

Automatically optimizes prompts by testing 3 variants and selecting the best:

```bash
python step3.py
```

Output files:
- `opro_train_iter1.json` - Variant 1 results
- `opro_train_iter2.json` - Variant 2 results
- `opro_train_iter3.json` - Variant 3 results
- `opro_test_results.json` - Test set evaluation
- `opro_summary.json` - Final summary with best variant

## Running All Steps

To run all steps sequentially:

```bash
python step1.py && python step2.py && python step3.py
```

## Dataset

The project uses **GSM8K (Grade School Math 8K)**, a dataset of 8.5K high-quality grade school math word problems requiring multi-step reasoning.

- Total questions: 8,500
- Sample size: 300 questions
- Train/test split for optimization: 70/30

Data is automatically downloaded and cached during the first run.

## Results

### Evaluation Metrics

- **Accuracy**: Percentage of questions answered correctly
- **Scoring method**: Exact match on final numeric answer
- Answer format: Final answer extracted from `####` prefix

### Expected Performance

Results vary based on LLM performance, but typically:
- Base prompt: ~40-50% accuracy
- Manual optimization: ~50-65% accuracy
- Automated optimization: ~55-70% accuracy

## Resuming Interrupted Runs

If the script is interrupted during execution, rerunning the same step will automatically resume from where it left off. Results are saved after each question to prevent data loss.

## Dependencies

See `requirements.txt` for full list:
- `openai` - OpenAI API client
- `datasets` - Hugging Face datasets library

## Notes

- Uses `gpt-5-mini` model for cost efficiency
- Each step saves results incrementally to local JSON files
- API calls are made for each question, so monitor your OpenAI usage
- Initial run will download the GSM8K dataset (~100MB)


