import random
import tqdm
import argparse
from simtom.simtom_hitom import *
import collections
import json
import concurrent.futures
import threading
import os
from datetime import datetime
from llm_utils import *
from new_decompose import TheoryOfMindSystem

DATA_DIR_TELL = '../data/hitom_tell.jsonl'
DATA_DIR_NO_TELL = '../data/hitom_no_tell.jsonl'

# random.seed(0)
def parse_hitom(file_path):
    """
    Parses HiToM data.

    :param file_path: Path to the JSONL file.
    :return: List of dictionaries representing each line in the file.
    """
    parsed_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    # Parse the line as JSON
                    data = json.loads(line.strip())
                    data["id"] = line_number  # Add a unique question ID
                    parsed_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return parsed_data

def start_task(llm,story, question, choices,note):
    story = "\n".join(story) if isinstance(story, list) else story
    choices = "\n".join(choices) if isinstance(choices, list) else choices

    prompt=f'''Read the following story and answer the multiple-choice question. Please provide answer without explanations.
    
    Story: {story}

    {question}
    Choices: {choices}

    {note}

    Answer with ONLY the correct choice. The answer should contain only a single word.

    Format: <option_letter>: <answer>

    Answer: 
'''
    print(prompt)
    answer = llm.get_output(prompt).strip().strip(".")
    return answer

def start_task_cot(llm,story, question, choices,note):
    story = "\n".join(story) if isinstance(story, list) else story
    choices = "\n".join(choices) if isinstance(choices, list) else choices

    prompt=f'''Read the following story and answer the multiple-choice question. Think step-by-step and then provide the answer.
    
    Story: {story}

    {question}
    Choices: {choices}

    {note}

    Provide the relevant label alongside the answer when providing your answer (<option_label>: <answer>).
    '''
    cot = llm.get_output(prompt)
    answer = llm.get_output(f'''This is the provided explanation for a question: {cot}
    Provide the answer selected in the above solution. Answer with ONLY the correct choice. The answer should contain only a single word.
    Format: <option_letter>: <answer> 
    Answer: ''').strip().strip(".")
    return answer

def evaluate_hitom():
    tell_data = parse_hitom(DATA_DIR_TELL)
    no_tell_data = parse_hitom(DATA_DIR_NO_TELL)
    language_model = LanguageModel(model_name = args.model, model_type = args.model_type)
    # Create results directory if not exists
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    # Create a unique log file name based on datetime
    log_filename = os.path.join(results_dir, f"hitom_{args.method}_{args.num_problems}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Group data into partitions, excluding order 0
    partitions = collections.defaultdict(list)
    for entry in tell_data + no_tell_data:
        descriptor = entry.get("descriptor", {})
        order = descriptor.get("order", 0)
        length = descriptor.get("length", 0)
        if order == 0:
            continue  # Exclude order 0
        type_ = "tell" if entry in tell_data else "no_tell"
        partitions[(order, length, type_)].append(entry)

    # Determine the number of problems to sample from each partition
    if args.num_problems > 0:
        total_partitions = len(partitions)
        problems_per_partition = args.num_problems // total_partitions
        selected_data = []

        for partition, entries in partitions.items():
            selected_data.extend(random.sample(entries, min(problems_per_partition, len(entries))))
    else:
        selected_data = [entry for partition in partitions.values() for entry in partition]

    categories = {
        "order": {},
        "length": {},
        "tell_no_tell": {"tell": {"correct": 0, "total": 0}, "no_tell": {"correct": 0, "total": 0}}
    }

    lock = threading.Lock()
    language_model = LanguageModel(model_name = args.model, model_type = args.model_type, temperature=0)

    detailed_logs = []  # List to store detailed logs

    def process_entry(language_model, entry, category_name, categories, questionPrompt):
        question = entry.get("question")
        options = {choice.split(". ")[1]: choice.split(". ")[0] for choice in entry.get("choices", [])}
        correct_answer = entry.get("answer").split()[1].strip()
        descriptor = entry.get("descriptor", {})

        order = descriptor.get("order")
        length = descriptor.get("length")
        answer_label = options[correct_answer].strip(".").strip().lower()
        if len(answer_label.split(" ")) > 1:
            answer_label = 'a'

        if order == 0:
            return

        questionP = questionPrompt.format(question=question, choices=options)
        if not args.random_example:
            with lock:
                if order not in categories["order"]:
                    categories["order"][order] = {"correct": 0, "total": 0}
                if length not in categories["length"]:
                    categories["length"][length] = {"correct": 0, "total": 0}
        system = TheoryOfMindSystem(mode = "hitom", model = args.model, model_type = args.model_type)
        if args.method == "decompose":
            story = "\n".join(entry.get("story", []))
            result = system.start_task(story, question, entry.get("choices", []), entry.get("note")).lower()
        elif args.method == "simtom":
            result, perspective = evalQuestion(language_model, entry, questionP, simModel=None)
            result = result.lower()
        elif args.method == "baseline":
            disamb = system.disambiguate_story(entry.get("story", []))
            story = disamb + entry.get("story", [])
            result = start_task(language_model, story, question, entry.get("choices", []), entry.get("note")).lower()
        elif args.method == "cot":
            disamb = system.disambiguate_story(entry.get("story", []))
            story = disamb + entry.get("story", [])
            result = start_task_cot(language_model, story, question, entry.get("choices", []), entry.get("note")).lower()
        if args.random_example:
            return result
        else:
            log_entry = {
                "id": entry["id"],
                "question": question,
                "correct_answer": f"{answer_label}: {correct_answer}",
                "returned_answer": result
            }
            with lock:
                with open(log_filename, "a") as log_file:
                    json.dump(log_entry, log_file)
                    log_file.write("\n")


            with lock:
                if (correct_answer in result) or (result.strip()[0] == answer_label):
                    categories["order"][order]["correct"] += 1
                    categories["length"][length]["correct"] += 1
                    categories["tell_no_tell"][category_name]["correct"] += 1
                categories["order"][order]["total"] += 1
                categories["length"][length]["total"] += 1
                categories["tell_no_tell"][category_name]["total"] += 1
    if args.random_example:
        random_entry = random.choice(selected_data)
        result = process_entry(language_model,random_entry,"","",questionPrompt)
        story_text = "\n".join(random_entry.get("story", []))
        choices_text = "\n".join(random_entry.get("choices", []))
        correct_answer = random_entry.get("answer")
        print(f"Question: {story_text} \nChoices: {choices_text} \n{correct_answer}\nGiven Result: {result}")
        return
    elif args.parallel_execution:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
            futures = []
            for entry in tqdm.tqdm(selected_data, desc="Processing Selected Data (Parallel)"):
                category_name = "tell" if entry in tell_data else "no_tell"
                futures.append(executor.submit(process_entry, language_model, entry, category_name, categories, questionPrompt))
            concurrent.futures.wait(futures)
    else:
        for entry in tqdm.tqdm(selected_data, desc="Processing Selected Data (Sequential)"):
            category_name = "tell" if entry in tell_data else "no_tell"
            process_entry(language_model, entry, category_name, categories, questionPrompt)

    for category, stats in categories.items():
        if category == "tell_no_tell":
            for key, values in stats.items():
                values["accuracy"] = (values["correct"] / values["total"] * 100) if values["total"] > 0 else 0
        else:
            for key, values in stats.items():
                values["accuracy"] = (values["correct"] / values["total"] * 100) if values["total"] > 0 else 0

    # Calculate overall accuracy
    overall_correct = sum([stats["correct"] for stats in categories["tell_no_tell"].values()])
    overall_total = sum([stats["total"] for stats in categories["tell_no_tell"].values()])
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print(overall_total)

    # Print results
    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"OVERALL ACCURACY: {overall_accuracy:.2f}%")
    print("\nAccuracy by Order:")
    for order, stats in categories["order"].items():
        print(f"  Order {order}: {stats['accuracy']:.2f}%")
    print("\nAccuracy by Length:")
    for length, stats in categories["length"].items():
        print(f"  Length {length}: {stats['accuracy']:.2f}%")
    print("\nAccuracy by Tell/No-Tell:")
    for category, stats in categories["tell_no_tell"].items():
        print(f"  {category.capitalize()}: {stats['accuracy']:.2f}%")
    print("------------------------\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument("--model", type=str, default="gpt-4o", help="Name of intended model to conduct analysis")
    parser.add_argument("--model_type", type=str, default="openai",choices=["openai", "gemini","local"],help="Select a model type.")
    parser.add_argument('--parallel_execution', action='store_true', help="Enable or disable parallel execution.")
    parser.add_argument('--random_example', action='store_true', help="Evaluate a single random example from the dataset.")
    parser.add_argument('--method', type=str, choices=['cot', 'baseline', 'simtom','decompose'], default='baseline', help="Method to use for evaluation.")
    parser.add_argument('--num_problems', type=int, default=0, help="Number of problems to evaluate.")
    parser.add_argument('--num_parallel', type=int, default=os.cpu_count(), help="Number of threads for parallel execution.")


    global args
    args = parser.parse_args()

    evaluate_hitom()

if __name__ == '__main__':
    main()
