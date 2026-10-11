
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def main():
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducible synthetic gene-expression-like data.
    X, y = make_classification(
        n_samples=120,
        n_features=30,
        n_informative=6,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.5, 0.5],
        class_sep=1.0,
        flip_y=0.05,
        random_state=42,
    )

    gene_names = [f"GENE_{i:02d}" for i in range(1, 31)]
    sample_names = [f"SAMPLE_{i:03d}" for i in range(1, 121)]

    expression = pd.DataFrame(
        X,
        index=sample_names,
        columns=gene_names,
    )
    expression.index.name = "sample_id"

    # Explicit labels: 0 = Control, 1 = Case.
    expression.insert(
        0,
        "class",
        np.where(y == 1, "Case", "Control"),
    )

    output_path = output_dir / "demo_expression.csv"
    expression.to_csv(output_path)

    print(f"Synthetic dataset saved to: {output_path}")
    print(f"Samples: {len(expression)}")
    print(f"Features: {len(gene_names)}")
    print("Class counts:")
    print(expression["class"].value_counts())
    print(
        "\nIMPORTANT: These are synthetic data, "
        "not measurements from real patients."
    )


if __name__ == "__main__":
    main()
