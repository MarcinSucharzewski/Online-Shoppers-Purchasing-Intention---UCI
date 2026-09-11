import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.train_model import (
    build_models,
    build_preprocessor,
    evaluate_model,
    find_f1_threshold,
)


def test_find_f1_threshold_returns_a_valid_probability():
    target = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.40, 0.55, 0.80, 0.95])

    threshold = find_f1_threshold(target, probabilities)

    assert 0.0 <= threshold <= 1.0


def test_model_pipeline_trains_and_returns_metrics():
    features = pd.DataFrame(
        {
            "pages": [1, 2, 3, 4, 5, 6, 7, 8],
            "visitor_type": ["New", "Returning"] * 4,
        }
    )
    target = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )
    models = build_models(build_preprocessor(x_train))

    model = models["logistic_regression"]
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test, threshold=0.5)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["confusion_matrix"]