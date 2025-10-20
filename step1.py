import re
import json
import random
import os
from datasets import load_dataset
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")

def download_gsm8k_data(cache_dir="./gsm8k_data", sample_size=300):
    print("Downloading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    all_data = list(train_data) + list(test_data)
    sampled_data = random.sample(all_data, min(sample_size, len(all_data)))
    
    print(f"Downloaded {len(all_data)} total questions, sampled {len(sampled_data)} for experiments")
    return sampled_data


def extract_final_answer(text):
    match = re.search(r'#### (\d+)', text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def extract_ground_truth(answer_text):
    return extract_final_answer(answer_text)


def score_response(llm_response, ground_truth_answer):
    predicted_answer = extract_final_answer(llm_response)
    
    if predicted_answer is None:
        return False
    
    return predicted_answer == ground_truth_answer


def create_base_prompt(question):
    return f"""{question}

Output the final answer at the end with the prefix ####. For example: #### ANSWER"""


def query_llm(prompt, model="gpt-4o-mini"):
    client = OpenAI(api_key=API_KEY)
    message = client.chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.choices[0].message.content


def evaluate_base_prompt(questions_data, results_file="evaluation_results.json"):
    test_data = questions_data
    total = len(test_data)
    results = {}
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} completed questions...")
    
    print(f"\nEvaluating base prompt on {total} questions...")
    print("=" * 60)
    
    for idx, item in enumerate(test_data, 1):
        if str(idx) in results:
            continue
        
        question = item['question']
        ground_truth = extract_ground_truth(item['answer'])
        
        if ground_truth is None:
            print(f"[{idx}/{total}] Could not extract ground truth, skipping")
            results[str(idx)] = {
                'idx': idx,
                'question': question,
                'is_correct': False,
                'predicted': None,
                'ground_truth': None,
                'llm_response': None
            }
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            continue
        
        prompt = create_base_prompt(question)
        
        try:
            llm_response = query_llm(prompt)
            is_correct = score_response(llm_response, ground_truth)
            
            if is_correct:
                status = "✓ Correct"
            else:
                predicted = extract_final_answer(llm_response)
                status = f"✗ Incorrect (predicted: {predicted}, ground truth: {ground_truth})"
            
            print(f"[{idx}/{total}] {status}")
            print(f"  Question: {question[:60]}...")
            print(f"  LLM response: {llm_response[:100]}...")
            print()
            
            results[str(idx)] = {
                'idx': idx,
                'question': question,
                'is_correct': is_correct,
                'predicted': extract_final_answer(llm_response),
                'ground_truth': ground_truth,
                'llm_response': llm_response
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"[{idx}/{total}] API error: {str(e)}")
            results[str(idx)] = {
                'idx': idx,
                'question': question,
                'is_correct': False,
                'predicted': None,
                'ground_truth': ground_truth,
                'llm_response': None,
                'error': str(e)
            }
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            continue
    
    correct = sum(1 for r in results.values() if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    print("=" * 60)
    print(f"\nBase prompt accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy


if __name__ == "__main__":
    data_file = 'gsm8k_sample_300.json'
    
    if os.path.exists(data_file):
        print("Loading existing sample data...")
        with open(data_file, 'r') as f:
            gsm8k_data = json.load(f)
        print(f"Loaded {len(gsm8k_data)} questions from {data_file}\n")
    else:
        print("Step 1: Download and sample data")
        gsm8k_data = download_gsm8k_data(sample_size=300)
        
        with open(data_file, 'w') as f:
            json.dump(gsm8k_data, f, indent=2)
        print(f"Data saved to {data_file}\n")
    
    print("Step 2: Evaluate with base prompt")
    base_accuracy = evaluate_base_prompt(gsm8k_data, results_file="base_prompt_results.json")
    
    print("\n✓ Step 1 complete!")
    print(f"Base accuracy: {base_accuracy:.2%}")
    print("Results saved to base_prompt_results.json")