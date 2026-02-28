from datasets import load_dataset
from config import CORPUS_PATH
import click


PROGRAMMING_KEYWORDS = [
    "python",
    "java",
    "c++",
    "javascript",
    "machine learning",
    "neural network",
    "deep learning",
    "algorithm",
    "data structure",
    "compiler",
    "operating system",
    "computer science",
    "source code",
    "software engineering"
]

def is_programming_related(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in PROGRAMMING_KEYWORDS)

@click.command()
@click.option(
    "--max-paragraphs",
    default=2000,
    type=int,
    show_default=True,
    help="Maximum number of programming-related paragraphs to extract"
)
def main(max_paragraphs):
    print(f"Loading Wiki40B dataset (target paragraphs: {max_paragraphs})...")

    dataset = load_dataset(
        "wiki40b",
        "en",
        split="train",
        streaming=True
    )

    paragraphs = []

    for article in dataset:
        text = article["text"]
        for para in text.split("\n"):
            para = para.strip()
            if not is_programming_related(para):
                continue
            if len(para.split()) > 40:
                paragraphs.append(para)
            if len(paragraphs) >= max_paragraphs:
                break
        if len(paragraphs) >= max_paragraphs:
            break
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for p in paragraphs:
            f.write(p.replace("\n", " ") + "\n")
    print(f"Saved {len(paragraphs)} paragraphs to {CORPUS_PATH}")

if __name__ == "__main__":
    main()
