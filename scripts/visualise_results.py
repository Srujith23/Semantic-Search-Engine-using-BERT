import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import textwrap

RESULTS_FILE = "results.json"
OUTPUT_IMAGE = "search_results.png"


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def wrap_text(text, width=55):
    """
    Wrap long text into multiple lines
    so it fits nicely inside graph nodes.
    """
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    data = load_results()

    query = data["query"]
    results = data["results"]

    # Larger canvas for full text
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # -------- Query Node --------
    query_x, query_y = 0.15, 0.5

    ax.text(
        query_x,
        query_y,
        f"Query:\n{query}",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.6",
            fc="#A6CEE3",
            ec="black"
        )
    )

    # -------- Result Nodes Layout --------
    n = len(results)

    # distribute results vertically
    ys = [0.85 - i*(0.7/(n-1 if n > 1 else 1)) for i in range(n)]

    for res, y in zip(results, ys):

        wrapped_doc = wrap_text(res["document"], width=55)

        result_text = (
            f"Rank {res['rank']} | Score {res['score']:.3f}\n\n"
            f"{wrapped_doc}"
        )

        result_x = 0.75

        ax.text(
            result_x,
            y,
            result_text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.6",
                fc="#B2DF8A",
                ec="black"
            )
        )

        # -------- Curved Arrow --------
        arrow = FancyArrowPatch(
            (query_x + 0.08, query_y),
            (result_x - 0.12, y),
            connectionstyle="arc3,rad=0.2",
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color="gray"
        )

        ax.add_patch(arrow)

    plt.title("Semantic Search Retrieval Results", fontsize=14, pad=20)
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE, dpi=220)
    print(f"Saved flow diagram to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()