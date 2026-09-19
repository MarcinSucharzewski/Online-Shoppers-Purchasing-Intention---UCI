# Online Shoppers Purchasing Intention

An end-to-end binary classification project that predicts whether an online shopping session will result in a purchase. The project uses the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) from the UCI Machine Learning Repository.

The repository contains a reusable Python training script, an exploratory Jupyter notebook, automated tests, model serialization, and generated evaluation reports.

## Project objective

The target variable is `Revenue`:

- `1` / `True` — the session ended with a purchase;
- `0` / `False` — the session did not end with a purchase.

The business objective is to rank active sessions by purchase probability. A model of this kind can support campaign targeting, personalized messages, session prioritization, and conversion analysis. It should be used as a decision-support system rather than as the sole basis for customer-related decisions.

## Dataset

The data is downloaded automatically with `ucimlrepo` using UCI dataset ID `468`:

```python
dataset = fetch_ucirepo(id=468)
```

The dataset used in the notebook contains:

| Item | Value |
| --- | ---: |
| Sessions | 12,330 |
| Input features | 17 |
| Missing values | 0 |
| Sessions without a purchase | 10,422 (84.5%) |
| Sessions with a purchase | 1,908 (15.5%) |

This is an imbalanced classification problem. A model that always predicts “no purchase” would obtain approximately 84.5% accuracy while failing to identify a single purchasing session. For this reason, accuracy is reported but is not used as the primary model-quality measure.

### Features

| Feature | Type loaded by the notebook | Description |
| --- | --- | --- |
| `Administrative` | Integer | Number of administrative pages visited during the session. |
| `Administrative_Duration` | Float | Total time spent on administrative pages. |
| `Informational` | Integer | Number of informational pages visited. |
| `Informational_Duration` | Float | Total time spent on informational pages. |
| `ProductRelated` | Integer | Number of product-related pages visited. |
| `ProductRelated_Duration` | Float | Total time spent on product-related pages. |
| `BounceRates` | Float | Average bounce-rate value of pages visited during the session. |
| `ExitRates` | Float | Average exit-rate value of pages visited during the session. |
| `PageValues` | Float | Average page value associated with pages viewed before a transaction. |
| `SpecialDay` | Float | Closeness of the session date to a special shopping day. |
| `Month` | Object | Month in which the session occurred. |
| `OperatingSystems` | Integer | Encoded operating-system identifier. |
| `Browser` | Integer | Encoded browser identifier. |
| `Region` | Integer | Encoded geographic-region identifier. |
| `TrafficType` | Integer | Encoded traffic-source category. |
| `VisitorType` | Object | Visitor category, such as new or returning visitor. |
| `Weekend` | Boolean | Indicates whether the session occurred during a weekend. |

`Revenue` is stored separately as the target and converted from text-like Boolean values to integers:

```python
target = target.astype(str).str.lower().map({"true": 1, "false": 0})
```

## Exploratory data analysis

The notebook performs the following analysis:

- displays the dataset dimensions, column types, missing-value counts, unique-value counts, and sample rows;
- calculates the distribution of the target class;
- separates columns by their pandas data types;
- plots distributions of the first nine numerical variables by purchase outcome;
- plots the most frequent values of categorical variables;
- calculates Pearson correlations between numerical features and `Revenue`;
- compares ROC and precision–recall curves;
- displays confusion matrices for both models;
- displays the 15 highest Random Forest feature importances.

In the stored notebook output, `PageValues` has the strongest linear correlation with purchase (`0.492569`). `ExitRates` has a negative correlation of `-0.207071`, and `BounceRates` has a negative correlation of `-0.150673`.

Correlation and feature importance describe associations in this dataset; they do not establish that a feature causes a purchase.

## Train/test split

The project uses an 80/20 split:

```python
x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target,
)
```

`stratify=target` preserves approximately the same purchase ratio in the training and test sets. `random_state=42` makes the split reproducible.

## Preprocessing

Preprocessing is implemented with a `ColumnTransformer` inside each scikit-learn pipeline. This ensures that preprocessing is fitted separately inside every training fold and then applied to validation or test data, reducing the risk of data leakage.

### Numerical pipeline

All columns whose pandas dtype is not `object` or `category` are assigned to the numerical pipeline:

1. missing values are replaced with the median using `SimpleImputer(strategy="median")`;
2. values are standardized using `StandardScaler()`.

With the dtypes shown in the notebook, this path includes the integer-coded `OperatingSystems`, `Browser`, `Region`, and `TrafficType` columns, as well as the Boolean `Weekend` column.

### Categorical pipeline

Columns with dtype `object` or `category` are assigned to the categorical pipeline:

1. missing values are replaced with the most frequent category;
2. categories are converted with `OneHotEncoder(handle_unknown="ignore")`.

In the downloaded data shown in the notebook, only `Month` and `VisitorType` enter this pipeline. Unknown categories encountered during prediction do not cause an error because they are ignored by the encoder.

## Models

The project compares two classifiers.

### Logistic Regression

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)
```

Logistic Regression is the linear baseline. `class_weight="balanced"` automatically gives more weight to the minority purchase class according to class frequency.

### Random Forest

The initial configuration is:

```python
RandomForestClassifier(
    class_weight="balanced",
    n_estimators=300,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)
```

Random Forest can model non-linear patterns and interactions between session variables. The training script uses all available CPU cores through `n_jobs=-1`; the notebook uses `n_jobs=1` for the search and cross-validation calls.

## Stratified cross-validation

Both model comparison and hyperparameter tuning use five-fold stratified cross-validation:

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

The training data is divided into five folds with similar class proportions. During each iteration, four folds are used for training and one for validation. Every fold is used once as the validation set. The project reports the mean and standard deviation of:

- ROC-AUC;
- average precision, labeled `PR-AUC` in the project.

## Random Forest hyperparameter tuning

`GridSearchCV` tests eight parameter combinations:

```python
param_grid = {
    "classifier__n_estimators": [200, 300],
    "classifier__max_depth": [None, 12],
    "classifier__min_samples_leaf": [1, 2],
}
```

Because eight combinations are evaluated with five folds, the search performs 40 cross-validation fits and then refits the best configuration on the full training set. ROC-AUC is the optimization metric.

The equivalent notebook uses the step name `model` instead of `classifier`, so its parameter names begin with `model__`.

The stored notebook selected:

```text
max_depth=None
min_samples_leaf=2
n_estimators=300
```

## F1-based threshold selection

The code does not assume that `0.5` is the best classification threshold. For each model, it generates out-of-fold purchase probabilities on the training set:

```python
out_of_fold_probabilities = cross_val_predict(
    model,
    x_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]
```

The precision–recall curve is calculated for these probabilities. The code computes the F1 score for every available threshold and selects the threshold with the highest F1 score:

```python
f1_scores = (2 * precision * recall / (precision + recall + 1e-12))[:-1]
best_threshold = thresholds[f1_scores.argmax()]
```

Using out-of-fold probabilities is important because each training observation is scored by a model that was fitted without that observation.

## Evaluation metrics

| Metric | Meaning |
| --- | --- |
| `ROC-AUC` | Measures how well predicted probabilities rank purchasing sessions above non-purchasing sessions across all thresholds. The project uses it as the principal ranking metric. |
| `PR-AUC` | Implemented with `average_precision_score`; summarizes precision–recall performance and is especially useful for an imbalanced positive class. |
| `F1` | Harmonic mean of precision and recall at the selected threshold. |
| `Balanced accuracy` | Average recall of the two classes, giving equal importance to purchases and non-purchases. |
| `Accuracy` | Overall fraction of correct predictions. It is included for context but can be misleading with this class distribution. |
| `Classification report` | Per-class precision, recall, F1 score, support, and aggregate averages. |
| `Confusion matrix` | Counts of true negatives, false positives, false negatives, and true positives. |

## Recorded notebook results

These values come from the outputs stored in the supplied notebook.

### Five-fold cross-validation on the training set

| Model | ROC-AUC mean | ROC-AUC std | PR-AUC mean | PR-AUC std |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9058 | 0.0054 | 0.6593 | 0.0172 |
| Random Forest | **0.9327** | **0.0036** | **0.7481** | **0.0168** |

### Held-out test set

| Model | ROC-AUC | PR-AUC | F1 | Balanced accuracy | Accuracy | Selected threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | **0.9253** | **0.7323** | **0.6716** | **0.8194** | **0.8917** | 0.4180 |
| Logistic Regression | 0.8962 | 0.6258 | 0.6313 | 0.7983 | 0.8759 | 0.5632 |

Random Forest achieved the higher ROC-AUC and PR-AUC in both cross-validation and the stored test evaluation.

### Leading Random Forest feature importances

The five highest importances stored in the notebook are:

| Transformed feature | Importance |
| --- | ---: |
| `numeric__PageValues` | 0.410348 |
| `numeric__ExitRates` | 0.092157 |
| `numeric__ProductRelated_Duration` | 0.088812 |
| `numeric__ProductRelated` | 0.063083 |
| `numeric__BounceRates` | 0.057811 |

Random Forest impurity-based importance indicates which variables the fitted trees used most. It does not show causal effects and can be biased toward features with more possible split points.

## Training script workflow

Running `src/train_model.py` performs the following operations:

1. downloads UCI dataset `468`;
2. converts `Revenue` to integer labels;
3. creates the stratified 80/20 train/test split;
4. builds the preprocessing pipelines and both classifiers;
5. tunes Random Forest with `GridSearchCV`;
6. calculates cross-validation ROC-AUC and PR-AUC statistics for both models;
7. selects an F1-maximizing threshold from out-of-fold training probabilities for each model;
8. refits each model on the full training set;
9. evaluates both models on the test set;
10. chooses the model with the highest test-set ROC-AUC;
11. saves the selected fitted pipeline with `joblib`;
12. writes a JSON report and a confusion-matrix image.

## Generated files

### `models/online_shoppers_model.joblib`

Contains the complete fitted pipeline: preprocessing plus the selected classifier. The saved pipeline can accept raw input columns with the same schema used during training.

### `reports/metrics.json`

Contains:

- dataset name, number of instances, and number of features;
- positive-class rate;
- test size, random seed, and number of CV folds;
- name of the selected model;
- mean and standard deviation of CV ROC-AUC and PR-AUC for each model;
- best Random Forest parameters;
- test ROC-AUC, PR-AUC, F1, balanced accuracy, accuracy, selected threshold, confusion matrix, and full classification report for each model.

### `reports/confusion_matrix.png`

Contains a confusion matrix for the model selected by test ROC-AUC.

Important implementation detail: `evaluate_model()` calculates the JSON metrics and confusion matrix using the optimized F1 threshold. The PNG is currently generated with `best_model.predict(x_test)`, which uses scikit-learn's default decision threshold of `0.5`. Therefore, the matrix in the PNG may differ from the matrix stored in `metrics.json`.

## Tests

`tests/test_train_model.py` contains two unit tests:

1. `test_find_f1_threshold_returns_a_valid_probability` verifies that the selected threshold is between `0.0` and `1.0`.
2. `test_model_pipeline_trains_and_returns_metrics` builds a small mixed-type dataset, trains the Logistic Regression pipeline, evaluates it, verifies that ROC-AUC and F1 are valid probabilities, and checks that a confusion matrix is returned.

The supplied Python files pass Python syntax compilation. The supplied notebook is valid notebook JSON.

## How to run

Python 3.10 or later is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/train_model.py
python -m pytest tests -q
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/train_model.py
python -m pytest tests -q
```

To open the exploratory analysis:

```bash
jupyter notebook analysis.ipynb
```

The code imports the following third-party packages: `joblib`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `seaborn`, and `ucimlrepo`. `pytest` is required to run the tests, and Jupyter is required to open the notebook interactively.

## Project structure

```text
.
├── analysis.ipynb
├── models/
│   └── online_shoppers_model.joblib      # created after training
├── reports/
│   ├── confusion_matrix.png              # created after training
│   └── metrics.json                      # created after training
├── src/
│   └── train_model.py
├── tests/
│   └── test_train_model.py
├── requirements.txt
└── README.md
```

## Reusing the saved model

```python
import json
import joblib

model = joblib.load("models/online_shoppers_model.joblib")

with open("reports/metrics.json", encoding="utf-8") as file:
    report = json.load(file)

threshold = report["models"][report["best_model"]]["threshold"]
purchase_probability = model.predict_proba(new_sessions)[:, 1]
purchase_prediction = (purchase_probability >= threshold).astype(int)
```

`new_sessions` must be a pandas DataFrame containing the same 17 feature names and compatible dtypes as the training data.

## Methodological notes and limitations

- The code chooses `best_model` by comparing ROC-AUC values on the held-out test set. For a stricter evaluation design, model selection should use cross-validation results, leaving the test set for one final unbiased estimate.
- Although `OperatingSystems`, `Browser`, `Region`, and `TrafficType` represent encoded categories, the current dtype-based preprocessor treats them as numerical variables and scales them. A future version could explicitly one-hot encode them and compare performance.
- The notebook performs the same train/test split twice. Because all parameters are identical, the second call reproduces the same split and does not change the results.
- Stored notebook results depend on the dataset version, package versions, random seed, and implementation. They should be regenerated after material code or dependency changes.
- Historical session behavior may not represent another shop or future customer behavior. Website changes, marketing changes, and shifts in traffic can cause model drift.
- The optimized F1 threshold assumes that precision and recall should be balanced. A production threshold should instead reflect the real costs of missed purchases and unnecessary interventions.
- Before deployment, the project would benefit from time-based validation, probability calibration, drift monitoring, and explicit data-privacy controls.

## Data source and citation

Sakar, C. O. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

The dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Refer to the UCI dataset page for the complete metadata, attribution requirements, and feature definitions.
