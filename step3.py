import re
import json
import random
import os
from datasets import load_dataset
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")

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


def evaluate_prompt(questions_data, prompt_template, results_file="evaluation_results.json"):
    test_data = questions_data
    total = len(test_data)
    results = {}
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} completed questions...")
    
    print(f"Evaluating on {total} questions...\n")
    
    for idx, item in enumerate(test_data, 1):
        if str(idx) in results:
            continue
        
        question = item['question']
        ground_truth = extract_ground_truth(item['answer'])
        
        if ground_truth is None:
            results[str(idx)] = {
                'idx': idx,
                'is_correct': False,
                'predicted': None,
                'ground_truth': None,
                'error': 'no_ground_truth'
            }
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            continue
        
        prompt = prompt_template.format(question=question)
        
        try:
            llm_response = query_llm(prompt)
            is_correct = score_response(llm_response, ground_truth)
            
            results[str(idx)] = {
                'idx': idx,
                'is_correct': is_correct,
                'predicted': extract_final_answer(llm_response),
                'ground_truth': ground_truth
            }
            
            status = "✓" if is_correct else "✗"
            print(f"[{idx}/{total}] {status} Completed")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            results[str(idx)] = {
                'idx': idx,
                'is_correct': False,
                'predicted': None,
                'ground_truth': ground_truth,
                'error': str(e)
            }
            print(f"[{idx}/{total}] ✗ Error: {str(e)[:50]}")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            continue
    
    correct = sum(1 for r in results.values() if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})\n")
    
    return accuracy


def get_prompt_variants():
    variants = {
        'variant_1': """Solve this math problem step by step.

{question}

Work through the problem carefully and show your reasoning.

IMPORTANT: Output the final numeric answer with ####. Format: #### [NUMBER]""",

        'variant_2': """You are a math tutor. Help solve this problem step by step.

{question}

Please:
1. Identify the key information
2. Determine what to find
3. Show all calculation steps
4. Provide the final answer

IMPORTANT: Write the final numeric answer with ####. Format: #### [NUMBER]""",

        'variant_3': """Solve this math word problem using chain of thought reasoning.

{question}

Think through each step:
- What information is given?
- What needs to be calculated?
- What operations are needed?
- Work through the calculation step by step
- State the final answer

IMPORTANT: End with the final numeric answer. Format: #### [NUMBER]"""
    }
    return variants


if __name__ == "__main__":
    data_file = 'gsm8k_sample_300.json'
    
    print("Loading sample data...")
    with open(data_file, 'r') as f:
        gsm8k_data = json.load(f)
    
    split_point = int(len(gsm8k_data) * 0.7)
    train_data = gsm8k_data[:split_point]
    test_data = gsm8k_data[split_point:]
    
    print(f"Loaded {len(gsm8k_data)} total questions")
    print(f"Train set: {len(train_data)} questions")
    print(f"Test set: {len(test_data)} questions\n")
    
    print("=" * 60)
    print("Step 3: Automated Prompt Engineering")
    print("=" * 60 + "\n")
    
    variants = get_prompt_variants()
    iteration_results = {}
    
    for iteration, (variant_name, prompt_template) in enumerate(variants.items(), 1):
        print(f"Iteration {iteration}: Testing {variant_name}")
        print("-" * 60)
        
        results_file = f"opro_train_iter{iteration}.json"
        if os.path.exists(results_file):
            os.remove(results_file)
        
        accuracy = evaluate_prompt(train_data, prompt_template, results_file=results_file)
        
        iteration_results[variant_name] = {
            'iteration': iteration,
            'accuracy': accuracy,
            'prompt': prompt_template
        }
        
        print(f"Iteration {iteration} ({variant_name}) accuracy: {accuracy:.2%}")
        print("=" * 60 + "\n")
    
    best_variant = max(iteration_results.items(), key=lambda x: x[1]['accuracy'])
    best_name = best_variant[0]
    best_accuracy = best_variant[1]['accuracy']
    best_prompt = best_variant[1]['prompt']
    
    print(f"✓ Best variant: {best_name}")
    print(f"  Train accuracy: {best_accuracy:.2%}\n")
    
    print("=" * 60)
    print("Evaluating best variant on test set")
    print("=" * 60 + "\n")
    
    test_results_file = "opro_test_results.json"
    if os.path.exists(test_results_file):
        os.remove(test_results_file)
    
    test_accuracy = evaluate_prompt(test_data, best_prompt, results_file=test_results_file)
    
    print(f"✓ Step 3 complete!")
    print(f"\nFinal Results:")
    print(f"  Best variant: {best_name}")
    print(f"  Train accuracy: {best_accuracy:.2%}")
    print(f"  Test accuracy: {test_accuracy:.2%}\n")
    
    summary = {
        'all_iterations': iteration_results,
        'best_variant': best_name,
        'train_accuracy': best_accuracy,
        'test_accuracy': test_accuracy,
        'best_prompt': best_prompt
    }
    
    with open('opro_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)