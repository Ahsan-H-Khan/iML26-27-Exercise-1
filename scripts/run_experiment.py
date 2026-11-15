"""Reproduce the prespecified two-dataset, three-seed experiment."""

from pathlib import Path

from iml_interpretability.data import load_benchmarks
from iml_interpretability.experiment import run_protocol


def main():
    output = Path("results/scores.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_protocol(load_benchmarks()).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
