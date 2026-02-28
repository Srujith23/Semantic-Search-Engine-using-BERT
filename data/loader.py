from config import CORPUS_PATH

def load_corpus(path=CORPUS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"""Specified file does not exist.......
                                __________WARNING PATH {path}___________""")
    documents = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                documents.append(text)
    return documents

if __name__=="__main__":
    docs = load_corpus()
    print(f"{len(docs)} Documents loaded successfully")