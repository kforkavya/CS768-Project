import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden channel size")
    parser.add_argument("--model", default="revgnn",
                        choices=["resgnn", "revgnn", "wt-resgnn", "wt-revgnn", "deq"],
                        help="Model type")
    parser.add_argument("--dataset", default="arxiv",
                        choices=["arxiv", "proteins"],
                        help="Dataset to use")
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    hidden = args.hidden
    save_path = f"plots/{model}_{hidden}ch_{dataset}_performance.png"
    layer_sizes = [3, 7, 14, 28]
    fig, ax = plt.subplots(figsize=(9, 5))
    for layer in layer_sizes:
        folder = f"results/{model}_{hidden}ch_{layer}L_{dataset}"
        file = f"{folder}/metrics.csv"
        df = pd.read_csv(file)
        ax.plot(df["epoch"], df["val_score"], label=f"{layer} layers", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Score")
    ax.set_title(f"Performance of {model} on {dataset}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()