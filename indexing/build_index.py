from data.loader import load_corpus
from utils.pooling import mean_pooling
from utils.preprocessing import preprocess
from model.encoder import BertSentenceEncoder
from config import INDEX_DIR, EMBEDDING_PATH, DOCS_PATH, RAW_DOCS_PATH
import torch

def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    raw_docs = load_corpus()
    docs = [preprocess(doc) for doc in raw_docs]
    encoder = BertSentenceEncoder()
    embeddings, attention_mask = encoder.encode(docs)
    pooled_embeddings = mean_pooling(embeddings, attention_mask)
    torch.save(pooled_embeddings, EMBEDDING_PATH)

    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(doc + '\n')
    print(f"Index built success for {len(docs)} documents!")

    with open(RAW_DOCS_PATH, 'w', encoding='utf-8') as f:
        for doc in raw_docs:
            f.write(doc + '\n')

if __name__=="__main__":
    build_index()