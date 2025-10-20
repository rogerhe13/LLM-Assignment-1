from datasets import load_dataset

dataset = load_dataset("gsm8k", "main", cache_dir='./data')


train_data = dataset['train']  
test_data = dataset['test']   


print(train_data[0])