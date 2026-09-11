# Online Shoppers Purchasing Intention

Classification project based on the [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) dataset from the UCI Machine Learning Repository.

## Project goal

The goal is to predict whether an online shopping session will end with a purchase (`Revenue`). This is an unbalanced binary classification problem: only 1,908 of 12,330 sessions end with a purchase.

The project shows a complete, reproducible process:

- automatic data download with `ucimlrepo`,
- a stratified train/test split,
- preprocessing for numeric and categorical features,
- a comparison of Logistic Regression and Random Forest,
- 5-fold stratified cross-validation,
- Random Forest hyperparameter tuning with `GridSearchCV`,
- F1-based threshold selection from out-of-fold predictions,
- evaluation with ROC-AUC, PR-AUC, F1, and balanced accuracy,
- the `analysis.ipynb` notebook with EDA, charts, ROC/PR curves, and feature interpretation,
- unit tests for the model pipeline and threshold logic,
- GitHub Actions workflow that runs the tests on every push and pull request,
- saving the best pipeline and a JSON report.

## Results

Results are generated locally when the script runs because the model downloads the data directly from UCI. The output files are saved in `reports/` and `models/`.

The main metric is `ROC-AUC`. Because the classes are unbalanced, we also check `F1`, `PR-AUC`, and the confusion matrix. This prevents the rare purchase class from being hidden by accuracy.

## How to run

Python 3.10+ is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train_model.py
python -m pytest tests -q
```

After the script finishes:

- `models/online_shoppers_model.joblib` contains the trained pipeline,
- `reports/metrics.json` contains the metrics and confusion matrix,
- `reports/confusion_matrix.png` contains the result visualization.

## Project structure

```text
.
├── models/                 # saved pipeline, created after training
├── reports/                # metrics and charts, created after training
├── analysis.ipynb           # analysis notebook with EDA and modeling
├── src/
│   └── train_model.py       # data download, training, tuning, and evaluation
├── tests/
│   └── test_train_model.py  # unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Business use

The model can identify sessions with a higher purchase probability. This can support personalized messages or marketing priorities. It should not be the only source for decisions: user behavior must be monitored and the cost of false alerts should be calibrated.

## Data source and citation

Sakar, C. O. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

The data is available under the CC BY 4.0 license. The full feature description is available on the UCI website.
