
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/demo_expression.csv")
OUTPUT_DIR = Path("results")
FIGURE_DIR = OUTPUT_DIR / "figures"
RANDOM_STATE = 42


def load_data(path):
    """Load and validate sample-level gene-expression data."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run "
            "'python src/generate_demo_data.py' first."
        )

    data = pd.read_csv(path, index_col=0)

    if "class" not in data.columns:
        raise ValueError("Dataset must contain a 'class' column.")

    if data.index.has_duplicates:
        raise ValueError("Duplicate sample identifiers found.")

    if data.empty:
        raise ValueError("Dataset is empty.")

    labels = data["class"].astype(str)
    if set(labels.unique()) != {"Control", "Case"}:
        raise ValueError(
            "The class column must contain exactly "
            "'Control' and 'Case'."
        )

    X = data.drop(columns="class").apply(
        pd.to_numeric, errors="raise"
    )

    if X.empty or X.shape[1] < 1:
        raise ValueError("No gene-expression features found.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Missing or non-finite feature values found.")

    if X.columns.has_duplicates:
        raise ValueError("Duplicate feature names found.")

    y = (labels == "Case").astype(int)

    if y.value_counts().min() < 5:
        raise ValueError(
            "At least five samples per class are required "
            "for this five-fold demonstration."
        )

    return X, y


def build_models():
    """Define models; scaling is fitted within each training fold."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_models(X, y):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    metric_rows = []
    prediction_table = pd.DataFrame(index=X.index)
    roc_data = {}

    # Out-of-fold predictions: each sample is held out once.
    for name, model in models.items():
        probabilities = np.zeros(len(y), dtype=float)
        predictions = np.zeros(len(y), dtype=int)

        for train_idx, test_idx in cv.split(X, y):
            fold_model = clone(model)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])

            probabilities[test_idx] = (
                fold_model.predict_proba(X.iloc[test_idx])[:, 1]
            )
            predictions[test_idx] = (
                probabilities[test_idx] >= 0.5
            ).astype(int)

        metric_rows.append({
            "model": name,
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(
                y, predictions, zero_division=0
            ),
            "recall": recall_score(
                y, predictions, zero_division=0
            ),
            "f1": f1_score(
                y, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y, probabilities),
        })

        prediction_table[f"{name}_probability"] = probabilities
        prediction_table[f"{name}_prediction"] = predictions

        fpr, tpr, _ = roc_curve(y, probabilities)
        roc_data[name] = (fpr, tpr, roc_auc_score(y, probabilities))

        # Out-of-fold confusion matrix.
        cm = confusion_matrix(y, predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Control", "Case"],
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"{name}: Out-of-Fold Confusion Matrix")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(
            FIGURE_DIR / f"{safe_name}_confusion_matrix.png",
            dpi=300,
        )
        plt.close()

    # ROC curves based on out-of-fold predictions.
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc_value) in roc_data.items():
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_value:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Out-of-Fold ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curves.png", dpi=300)
    plt.close()

    # Save metrics and per-sample out-of-fold predictions.
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    prediction_table.insert(0, "true_class", y.map({
        0: "Control",
        1: "Case",
    }))
    prediction_table.to_csv(
        OUTPUT_DIR / "out_of_fold_predictions.csv"
    )

    # Fit on all samples for exploratory feature rankings only.
    # These rankings are not independent validation results.
    for name, model in models.items():
        full_model = clone(model)
        full_model.fit(X, y)

        if name == "Logistic Regression":
            estimator = full_model.named_steps["logisticregression"]
            importance = np.abs(estimator.coef_[0])
            importance_label = "Absolute standardized coefficient"
        else:
            importance = full_model.feature_importances_
            importance_label = "Random forest importance"

        ranking = pd.DataFrame({
            "gene": X.columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        ranking.to_csv(
            OUTPUT_DIR /
            f"{name.lower().replace(' ', '_')}_feature_ranking.csv",
            index=False,
        )

        top = ranking.head(15).sort_values("importance")
        plt.figure(figsize=(8, 6))
        plt.barh(top["gene"], top["importance"])
        plt.xlabel(importance_label)
        plt.title(f"{name}: Exploratory Feature Ranking")
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR /
            f"{name.lower().replace(' ', '_')}_feature_importance.png",
            dpi=300,
        )
        plt.close()

    print("\nCross-validated model metrics:")
    print(metrics.round(3).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")
    print(
        "\nInterpretation: synthetic-data demonstration only. "
        "Feature rankings from models fitted to all samples are "
        "exploratory, not independently validated biomarkers."
    )


def main():
    X, y = load_data(DATA_PATH)
    print(f"Loaded {X.shape[0]} samples and {X.shape[1]} features.")
    evaluate_models(X, y)


if __name__ == "__main__":
    main()
