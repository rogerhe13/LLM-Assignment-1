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


def create_improved_prompt(question):
    return f"""You are an expert math problem solver. Solve the following problem step by step.

Problem: {question}

Please work through this problem carefully:
1. Identify what information is given
2. Determine what needs to be found
3. Plan your solution approach
4. Execute each step clearly
5. Verify your answer

IMPORTANT: You MUST output the final numeric answer at the end with the prefix ####.
Format: #### [NUMBER]
Example: If the answer is 42, write: #### 42"""


def create_cot_prompt(question):
    return f"""Solve this math problem step by step using chain of thought reasoning.

{question}

Think through each step:
- What do I know?
- What do I need to find?
- What operations should I perform?
- What is my final answer?

IMPORTANT: You MUST output the final numeric answer at the end with the prefix ####.
Format: #### [NUMBER]
Example: If the answer is 42, write: #### 42"""


def query_llm(prompt, model="gpt-5-mini"):
    client = OpenAI(api_key=API_KEY)
    message = client.chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.choices[0].message.content


def evaluate_prompt(questions_data, prompt_func, results_file="evaluation_results.json"):
    test_data = questions_data
    total = len(test_data)
    results = {}
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} completed questions...")
    
    print(f"\nEvaluating on {total} questions...")
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
        
        prompt = prompt_func(question)
        
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
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy


if __name__ == "__main__":
    data_file = 'gsm8k_sample_300.json'
    
    print("Loading sample data...")
    with open(data_file, 'r') as f:
        gsm8k_data = json.load(f)
    print(f"Loaded {len(gsm8k_data)} questions\n")
    
    print("Step 2a: Evaluate improved prompt")
    improved_accuracy = evaluate_prompt(gsm8k_data, create_improved_prompt, 
                                        results_file="improved_prompt_results.json")
    
    print("\n" + "=" * 60)
    print("Step 2b: Evaluate chain of thought prompt")
    cot_accuracy = evaluate_prompt(gsm8k_data, create_cot_prompt, 
                                   results_file="cot_prompt_results.json")
    
    print("\n✓ Step 2 complete!")
    print(f"Improved prompt accuracy: {improved_accuracy:.2%}")
    print(f"Chain of thought accuracy: {cot_accuracy:.2%}")