import random
import tqdm
import argparse
import collections
import json
import concurrent.futures
import threading
import os
from datetime import datetime
from llm_utils import *
from simtom.simtom_fantom import *
from new_decompose import TheoryOfMindSystem

def parse_dataset(file_path):
    """
    Parses the dataset JSONL file.
    :param file_path: Path to the dataset file.
    :return: Parsed list of entries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            entry = json.loads(line.strip())
            entry['id'] = line_number  # Add a unique question ID
            data.append(entry)
    return data

def generate_choices(correct_answer, wrong_answer):
    """
    Randomly assign correct and wrong answers to options with labels.
    """
    answer_goes_last = random.choice([True, False])
    if answer_goes_last:
        choices = [wrong_answer, correct_answer]
        correct_index = 1
    else:
        choices = [correct_answer, wrong_answer]
        correct_index = 0

    option_letters = [chr(x) for x in range(ord('a'), len(choices) + ord('a'))]
    choices_text = "\n".join(f"({letter}) {choice}" for letter, choice in zip(option_letters, choices))
    return choices_text, correct_index, option_letters

def start_task(llm, story, question, choices, note):
    """
    Start a task using a language model to answer a multiple-choice question.
    """
    prompt = f'''Read the following sequence of dialogues and answer the multiple-choice question. Provide your answer without explanations.

    Story: {story}

    {question}
    Choices:
    {choices}

    {note}

    Answer with ONLY the correct choice. Format: (option_letter)

    Answer: 
    '''
    return llm.get_output(prompt).strip().strip(".")

def start_task_cot(llm, story, question, choices, note):
    """
    Start a task using a language model to answer a multiple-choice question.
    """
    prompt = f'''Read the following sequence of dialogues and answer the multiple-choice question. Think step-by-step and then provide the answer.

    Story: {story}

    {question}
    Choices:
    {choices}

    Provide the relevant label alongside the answer when providing your answer
    '''
    cot = llm.get_output(prompt)
    answer = llm.get_output(f'''This is the provided explanation for a question: {cot}
    Provide the answer selected in the above solution. Answer with ONLY the correct choice. The answer should contain only a single word.
    Format: (option_letter)
    Answer: ''').strip().strip(".")
    return answer

def evaluate_dataset(file_path, method, num_problems, context, parallel_execution, num_parallel, model, model_type):
    """
    Evaluate a dataset based on specified methods and parameters.
    """
    data = parse_dataset(file_path)

    # Prepare results directory
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    log_filename = os.path.join(results_dir, f"fantom_{method}_{num_problems}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Sampling logic
    if num_problems > 0:
        sampled_data = data[:min(num_problems, len(data))]
    else:
        sampled_data = data

    # Progress logging
    correct_count = 0
    total_count = 0
    detailed_logs = []
    lock = threading.Lock()

    # Model initialization
    language_model = LanguageModel(model, temperature=0, model_type=model_type)

    def process_entry(entry):
        nonlocal correct_count, total_count
        story = entry[context+"_context"]
        question = entry["question"]
        correct_answer = entry["correct_answer"]
        wrong_answer = entry["wrong_answer"]
        note = entry.get("note", "")

        # Generate randomized choices
        choices_text, correct_index, option_letters = generate_choices(correct_answer, wrong_answer)
        system = TheoryOfMindSystem(mode = "fantom", model = model, model_type = model_type)
        # Select method
        if method == "baseline":
            returned_answer = start_task(language_model, story, question, choices_text, note)
        elif method == "cot":
            returned_answer = start_task_cot(language_model, story, question, choices_text, note)
        elif method == "simtom":
            returned_answer,_ = evalQuestion(language_model, story, questionPrompt.format(question = question, choices = choices_text),  simModel=None)
        elif method == "decompose":
            returned_answer = system.start_task(story, question, choices_text, "").lower()
        # Check correctness
        if "Answer:" in returned_answer:
            returned_answer=returned_answer.split("Answer:")[1]
        response = returned_answer.lower().strip().strip("**")
        if response=="":
            return
        label = "("+option_letters[correct_index].lower()+")"
        answer =  option_letters[correct_index]
        is_correct = response.startswith("(" + answer + ")") or response.startswith(answer + ")") or response.startswith(answer + ".") or response.startswith(answer + ":") or response.startswith(answer + ",") or "({})".format(answer) in response or answer == response
        if is_correct:
            correct_count += 1
        total_count += 1

        # Save log entry
        log_entry = {
            "id": entry["id"],
            "question": question,
            "choices": choices_text.strip(),
            "correct_answer": option_letters[correct_index] +" " + correct_answer,
            "returned_answer": returned_answer,
            "is_correct": is_correct
        }
        with lock:
            with open(log_filename, "a") as log_file:
                json.dump(log_entry, log_file)
                log_file.write("\n")

    # Parallel or sequential execution
    if parallel_execution:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
            list(tqdm.tqdm(executor.map(process_entry, sampled_data), total=len(sampled_data), desc="Evaluating"))
    else:
        for entry in tqdm.tqdm(sampled_data, desc="Evaluating"):
            process_entry(entry)


    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Evaluation complete! Results saved to {log_filename}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count} correct)")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a dataset with specified context and methods.")
    parser.add_argument("--file", type=str, default="../data/fantomtom.jsonl", help="Path to the dataset JSONL file.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Name of intended model to conduct analysis")
    parser.add_argument("--model_type", type=str, default="openai",choices=["openai", "gemini","local"],help="Select a model type.")
    parser.add_argument("--method", type=str, choices=["baseline", "cot","simtom","decompose"], required=True, help="Evaluation method.")
    parser.add_argument("--num_problems", type=int, default=0, help="Number of problems to evaluate (0 for all).")
    parser.add_argument("--context", type=str, choices=["short", "full"], required=False, default = "short", help="Context type to use.")
    parser.add_argument('--parallel_execution', action='store_true', help="Enable or disable parallel execution.")
    parser.add_argument("--num_parallel", type=int, default=os.cpu_count(), help="Number of threads for parallel execution.")
    args = parser.parse_args()

    evaluate_dataset(args.file, args.method, args.num_problems, args.context, args.parallel_execution, args.num_parallel, args.model, args.model_type)

if __name__ == "__main__":
    main()
