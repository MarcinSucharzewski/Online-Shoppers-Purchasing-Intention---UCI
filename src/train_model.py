"""Train and evaluate purchase-intention classifiers on the UCI dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "online_shoppers_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "reports" / "confusion_matrix.png"
RANDOM_STATE = 42
CV_SPLITS = 5


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Download the official UCI dataset and return features and target."""
    dataset = fetch_ucirepo(id=468)
    features = dataset.data.features.copy()
    target = dataset.data.targets.squeeze().copy()
    target = target.astype(str).str.lower().map({"true": 1, "false": 0})
    return features, target.astype("int64")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object", "category"]).columns
    numeric_columns = features.select_dtypes(exclude=["object", "category"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=300,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def find_f1_threshold(target: pd.Series, probabilities) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    f1_scores = (2 * precision * recall / (precision + recall + 1e-12))[:-1]
    return round(float(thresholds[f1_scores.argmax()]), 4)


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> dict:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions).tolist()
    return {
        "roc_auc": round(roc_auc_score(target, probabilities), 4),
        "pr_auc": round(average_precision_score(target, probabilities), 4),
        "f1": round(f1_score(target, predictions), 4),
        "balanced_accuracy": round(balanced_accuracy_score(target, predictions), 4),
        "accuracy": round(accuracy_score(target, predictions), 4),
        "threshold": threshold,
        "confusion_matrix": matrix,
        "classification_report": classification_report(
            target, predictions, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = build_models(build_preprocessor(x_train))
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    random_forest_search = GridSearchCV(
        estimator=models["random_forest"],
        param_grid={
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [1, 2],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    random_forest_search.fit(x_train, y_train)
    models["random_forest"] = random_forest_search.best_estimator_

    results = {}
    cv_results = {}
    thresholds = {}
    for name, model in models.items():
        cv_scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
            n_jobs=-1,
        )
        cv_results[name] = {
            "roc_auc_mean": round(float(cv_scores["test_roc_auc"].mean()), 4),
            "roc_auc_std": round(float(cv_scores["test_roc_auc"].std()), 4),
            "pr_auc_mean": round(float(cv_scores["test_pr_auc"].mean()), 4),
            "pr_auc_std": round(float(cv_scores["test_pr_auc"].std()), 4),
        }
        out_of_fold_probabilities = cross_val_predict(
            model,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        thresholds[name] = find_f1_threshold(y_train, out_of_fold_probabilities)
        model.fit(x_train, y_train)
        results[name] = evaluate_model(model, x_test, y_test, thresholds[name])

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = models[best_name]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    report = {
        "dataset": "UCI Online Shoppers Purchasing Intention Dataset",
        "instances": int(len(features)),
        "features": int(features.shape[1]),
        "positive_class_rate": round(float(target.mean()), 4),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "best_model": best_name,
        "cross_validation": cv_results,
        "best_random_forest_params": random_forest_search.best_params_,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions = best_model.predict(x_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No purchase", "Purchase"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(f"Confusion matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()

    print(f"Best model: {best_name}")
    print(f"ROC-AUC: {results[best_name]['roc_auc']:.4f}")
    print(f"F1: {results[best_name]['f1']:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
