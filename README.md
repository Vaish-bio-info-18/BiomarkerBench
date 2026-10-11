
# BiomarkerBench 🧬

**A reproducible Python toolkit for evaluating candidate biomarkers using gene-expression data and machine learning.**

## Overview

BiomarkerBench is a small computational biology project that compares classification models on gene-expression-like data. It demonstrates a reproducible workflow for model evaluation, prediction analysis, and exploratory feature ranking.

The repository includes a synthetic demo dataset generator, two machine-learning models, stratified cross-validation, ROC analysis, and automated result export.

## Features

- Input validation for sample-level expression matrices
- Logistic regression and random forest classification
- Five-fold stratified cross-validation
- Out-of-fold predictions for each sample
- ROC curves and ROC-AUC evaluation
- Accuracy, precision, recall, and F1-score
- Confusion matrix visualizations
- Exploratory feature-importance rankings
- Exportable CSV metrics and prediction tables

## Technologies

- Python
- NumPy and pandas
- scikit-learn
- Matplotlib

## Repository Structure

```text
biomarkerbench/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── demo_expression.csv
├── src/
│   ├── generate_demo_data.py
│   └── evaluate_biomarkers.py
└── results/
    ├── model_metrics.csv
    ├── out_of_fold_predictions.csv
    ├── logistic_regression_feature_ranking.csv
    ├── random_forest_feature_ranking.csv
    └── figures/
```

Results are generated when the scripts run and need not be committed to the repository.

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

## Run the Project

Generate the synthetic dataset:

```bash
python src/generate_demo_data.py
```

Evaluate the models:

```bash
python src/evaluate_biomarkers.py
```

The analysis generates model metrics, out-of-fold predictions, ROC curves, confusion matrices, and exploratory feature rankings in the `results/` directory.

## Evaluation Strategy

The project uses five-fold stratified cross-validation. Scaling for logistic regression is performed within each training fold to avoid leakage from held-out samples.

ROC-AUC and other classification metrics are calculated from out-of-fold predictions. Feature rankings are generated from models fitted to the complete dataset and are therefore exploratory rather than independently validated.

## Data Disclaimer

The included demo data are synthetic and were generated for software demonstration. They do not represent real patients, a specific disease, or experimentally validated biomarkers.

High performance on synthetic data must not be interpreted as evidence of clinical utility.

## Future Improvements

- Support for public gene-expression datasets
- Nested cross-validation for model selection
- Confidence intervals for performance metrics
- External validation on an independent cohort
- Differential expression analysis integration
- Improved interpretation of feature importance

## Author


