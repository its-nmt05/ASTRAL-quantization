from collections import Counter
from datasets import load_dataset
from tqdm import tqdm 
import json 

def load_data():
    dataset = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train")
    dataset.set_format(type="torch", columns=["audio", "transcription"])
    return dataset

def top_k(dataset, k=200):
    all_text = "".join(item["transcription"] for item in dataset)
    freq = Counter(all_text)
    sentence_scores = []
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing sentences"):
        score = sum(1 / freq[char] for char in set(item["transcription"]))
        sentence_scores.append((i, score))

    sentence_scores.sort(key=lambda x: -x[1])
    most_diverse_indices = [i for i, _ in sentence_scores[:k]]
    return most_diverse_indices

dataset = load_data()
indices = top_k(dataset)
with open('indices.json', 'w') as f:
    json.dump(indices, f)



