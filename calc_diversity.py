from datasets import load_dataset

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import fasttext
import re
import json
import random


MODEL_PATH = 'models/cc.vi.300.bin'
FILE_PATH = 'indices.json'
model = fasttext.load_model(MODEL_PATH)
N = 500


def preprocess_text(text):
    # Remove any unwanted characters or punctuation
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    return words


def get_average_embedding(text, model):
    words = preprocess_text(text)
    embeddings = [model.get_word_vector(word) for word in words if word]
    if len(embeddings) == 0:
        return np.zeroes(model.get_dimension())
    return np.mean(embeddings, axis=0)


def calc_diverse(dataset, N=-1):
    indices = list(range(len(dataset)))
    embeddings = np.array([get_average_embedding(dataset[i]['transcription'], model) for i in indices])
    similarity_matrix = cosine_similarity(embeddings)
    dissimilarity_matrix = 1 - similarity_matrix
    total_dissimilarity = np.sum(dissimilarity_matrix, axis=1)
    top_n = np.argsort(total_dissimilarity)[::-1][:N]
    return top_n.astype(int).tolist()


dataset = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train")
# dataset = dataset.select(range(samples))
dataset.set_format(type="torch", columns=["audio", "transcription"])
top_n = calc_diverse(dataset, N)

with open(FILE_PATH, 'w') as f:
    json.dump(top_n, f)
    