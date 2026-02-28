import torch
import numpy as np
import json

from model.encoder import BertSentenceEncoder
from utils.pooling import mean_pooling
from utils.preprocessing import preprocess
from config import EMBEDDING_PATH, TOP_K, RAW_DOCS_PATH


# ---------- Loading ----------
def load_embs():
    return torch.load(EMBEDDING_PATH, map_location="cpu")


def load_corpus():
    with open(RAW_DOCS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_index():
    return load_embs(), load_corpus()


# ---------- Embedding ----------
def query_embd(query, model):
    query = preprocess(query)
    embd, attention_mask = model.encode([query])
    final_embd = mean_pooling(embd, attention_mask)
    return final_embd.squeeze(0)


# ---------- Ranking ----------
def rank(query_embd, embs):
    scores = torch.nn.functional.cosine_similarity(
        query_embd.unsqueeze(0),
        embs,
        dim=1
    )

    scores = np.array(scores)
    top_k_indices = np.argsort(scores)[::-1][:TOP_K]
    top_k_scores = scores[top_k_indices]

    return top_k_indices, top_k_scores


# ---------- Search ----------
def search(query, model, embs, corpus):
    query_emb = query_embd(query, model)
    top_indices, top_scores = rank(query_emb, embs)

    results = [
        (corpus[i], float(score))
        for i, score in zip(top_indices, top_scores)
    ]
    return results


# ---------- Save JSON ----------
def save_results_json(query, results, filename="results.json"):
    data = {
        "query": query,
        "top_k": len(results),
        "results": []
    }

    for rank, (doc, score) in enumerate(results, start=1):
        data["results"].append({
            "rank": rank,
            "score": score,
            "document": doc
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---------- Main ----------
if __name__ == "__main__":
    model = BertSentenceEncoder()

    query = input("Enter query: ")
    embs, corpus = load_index()

    results = search(query, model, embs, corpus)

    print(f"\nTop {TOP_K} results:\n")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. ({score:.4f}) {doc}\n")

    save_results_json(query, results)
    print("Results saved to results.json")