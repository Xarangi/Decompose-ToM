# Decompose-ToM

This is the repository for the paper **[Decompose-ToM: Enhancing Theory of Mind Reasoning in Large Language Models through Simulation and Task Decomposition](https://arxiv.org/abs/2501.09056)**.  

## Setup

**Install dependencies**

Navigate to the `code` directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running Evaluation

There are two main evaluation scripts:


### 1. HiToM Evaluation

```bash
python evaluate_hitom.py [--category CATEGORY] [--model MODEL] [--model_type {openai,gemini,local}] [--parallel_execution] [--random_example] [--method {cot,baseline,simtom,decompose}] [--num_problems N] [--num_parallel N]
```

**Key arguments:**
- `--category`: Category to evaluate (default: all)
- `--model`: Model name (default: gpt-4o)
- `--model_type`: Model type (`openai`, `gemini`, or `local`; default: openai)
- `--parallel_execution`: Enable parallel execution
- `--random_example`: Evaluate a single random example
- `--method`: Evaluation method (`cot`, `baseline`, `simtom`, `decompose`; default: baseline)
- `--num_problems`: Number of problems to evaluate (default: 0 = all)
- `--num_parallel`: Number of threads for parallel execution (default: all CPU cores)

### 2. FanToM Evaluation

```bash
python evaluate_fantom.py [--model MODEL] [--model_type {openai,gemini,local}] --method {baseline,cot,simtom,decompose} [--num_problems N] [--context {short,full}] [--parallel_execution] [--num_parallel N]
```

**Key arguments:**
- `--file`: Path to the dataset JSONL file (default: ../data/fantomtom.jsonl)
- `--model`: Model name (default: gpt-4o)
- `--model_type`: Model type (`openai`, `gemini`, or `local`; default: openai)
- `--method`: Evaluation method (**required**: `baseline`, `cot`, `simtom`, `decompose`)
- `--num_problems`: Number of problems to evaluate (default: 0 = all)
- `--context`: Context type (`short` or `full`; default: short)
- `--parallel_execution`: Enable parallel execution
- `--num_parallel`: Number of threads for parallel execution (default: all CPU cores)

## Notes

- The scripts use OpenAI and Google Gemini APIs. Make sure to set your API keys as environment variables:
  - `OPENAI_API_KEY` for OpenAI
  - `GEMINI_API_KEY` for Gemini (Google Generative AI)
- You can change model settings in `llm_utils.py` or via script arguments.

## Folder Structure

- `evaluate_hitom.py` / `evaluate_fantom.py`: Main evaluation scripts
- `llm_utils.py`: Language model utility functions
- `new_decompose.py`: Core ToM system logic
- `simtom/`, `prompts/`: Supporting modules and prompt templates

The TheoryOfMindSystem class contains the Decompose-ToM method and can be configured to be run with different datasets.

