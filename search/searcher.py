import torch
from model.encoder import BertSentenceEncoder
from utils.pooling import mean_pooling
from utils.preprocessing import preprocess
from config import EMBEDDING_PATH, DOCS_PATH, TOP_K, RAW_DOCS_PATH
import numpy as np

def load_embs():
    embs = torch.load(EMBEDDING_PATH, map_location="cpu")
    return embs

def load_corpus():
    with open(RAW_DOCS_PATH, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
    
def query_embd(query, model):
    query = preprocess(query)
    embd, attention_mask = model.encode([query])
    final_embd = mean_pooling(embd, attention_mask)
    return final_embd.squeeze(0)

def rank(query_embd, embs):
    scores = torch.nn.functional.cosine_similarity(query_embd.unsqueeze(0), embs, dim=1)
    scores = np.array(scores)
    top_k_indices = np.argsort(scores)[::-1][:TOP_K]
    top_k_scores = scores[top_k_indices]
    return top_k_indices, top_k_scores

def load_index():
    embs = load_embs()
    corpus = load_corpus()
    return embs, corpus

def search(query, model, embs, corpus):
    query_emb = query_embd(query, model)
    top_indices, top_scores = rank(query_emb, embs)
    results = [
        (corpus[i], float(score)) 
         for i,score in zip(top_indices, top_scores)]
    return results

if __name__=="__main__":
    model = BertSentenceEncoder()
    query = input("Enter query: ")
    embs, corpus = load_index()
    results = search(query, model, embs, corpus)
    print(f'\n{TOP_K} results: ')
    for doc in results:
        print("-",doc)