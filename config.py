import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR/'data'
INDEX_DIR = BASE_DIR/'index'
EMBEDDING_PATH = INDEX_DIR/'embeddings.pt'
DOCS_PATH = INDEX_DIR/'docs.txt'
RAW_DOCS_PATH = INDEX_DIR/'raw_docs.txt'
CORPUS_PATH = DATA_DIR / "corpus.txt"

MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE=32

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_K = 3