import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


def load_band_features(csv_path, label_column="label"):
    """Load EEG windowed band features from a CSV file.

    The CSV should contain one row per window and one column per feature.
    The `label_column` contains the binary target for concentration vs non-concentration.
    """
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    X = df.drop(columns=[label_column])
    y = df[label_column].astype(int)

    metadata_cols = [
        "window_start_row",
        "window_end_row",
        "window_size_samples",
        "window_midpoint_sec",
        "window_start_sec",
        "window_end_sec",
    ]
    X = X.drop(columns=[c for c in metadata_cols if c in X.columns])
    return X, y


def replace_outliers(X, z_thresh=3.0, fill_method="median"):
    """Replace or clip outlier values in each feature column.

    We use a robust statistic (median absolute deviation) because EEG band
    feature distributions can have heavy tails and transient spikes.
    """
    Xr = X.copy()
    med = Xr.median()
    mad = Xr.mad().replace(0, np.finfo(float).eps)

    z = np.abs((Xr - med) / mad)
    mask = z > z_thresh

    if fill_method == "median":
        Xr[mask] = np.broadcast_to(med.values, Xr.shape)[mask.values]
    elif fill_method == "clip":
        lower = med - z_thresh * mad
        upper = med + z_thresh * mad
        Xr = Xr.clip(lower=lower, upper=upper, axis=1)
    else:
        raise ValueError("fill_method must be 'median' or 'clip'")

    return Xr


def smooth_windows(X, window_size=3, min_periods=1):
    """Smooth features across adjacent windows using a rolling mean.

    This makes the input less sensitive to isolated short-lived fluctuations.
    """
    return (
        X.rolling(window=window_size, min_periods=min_periods, center=True)
        .mean()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )


def build_pipeline(use_pca=True, pca_variance=0.95, scaler="standard", probability=True):
    """Build the sklearn pipeline for scaling, optional PCA, and SVM."""
    steps = []

    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    else:
        raise ValueError("scaler must be 'standard' or 'robust'")

    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))

    steps.append(("svc", SVC(kernel="rbf", probability=probability)))
    return Pipeline(steps)


def tune_and_train(X, y, use_pca=True, scaler="standard", probability=True):
    """Train the pipeline with grid search and return the best estimator."""
    pipe = build_pipeline(use_pca=use_pca, scaler=scaler, probability=probability)
    param_grid = {
        "svc__C": [0.1, 1.0, 10.0],
        "svc__gamma": ["scale", "auto"],
    }
    search = GridSearchCV(
        pipe,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    return search


def evaluate_model(model, X_test, y_test):
    """Print classification metrics for the test set."""
    y_pred = model.predict(X_test)
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def save_model(model, path):
    """Persist the trained model to disk."""
    joblib.dump(model, path)


def load_model(path):
    """Load a saved sklearn model or pipeline from disk."""
    return joblib.load(path)


def predict_focus(model, X, threshold=0.5, label_map=None):
    """Predict focus state and return confidence scores."""
    if label_map is None:
        label_map = {1: "focused", 0: "unfocused"}

    X_prepared = prepare_features(X, z_thresh=3.5, smooth_window=3, fill_method="median")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support probability estimates for confidence scoring")

    proba = model.predict_proba(X_prepared)
    if proba.shape[1] != 2:
        raise ValueError("Expected binary classification probabilities")

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Expected labels containing 1 for the focused state")

    positive_index = classes.index(1)
    positive_probs = proba[:, positive_index]
    labels = [label_map[1] if p >= threshold else label_map[0] for p in positive_probs]

    result = pd.DataFrame({"confidence": positive_probs, "focus": labels}, index=X.index)
    return result


def main(csv_path, label_column="label", save_model_path=None):
    X, y = load_band_features(csv_path, label_column=label_column)
    X_prepared = prepare_features(X, z_thresh=3.5, smooth_window=3, fill_method="median")

    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model_search = tune_and_train(X_train, y_train, use_pca=True, scaler="standard", probability=True)
    print("Best parameters:", model_search.best_params_)
    print("Best cross-validation accuracy:", model_search.best_score_)

    best_model = model_search.best_estimator_
    evaluate_model(best_model, X_test, y_test)

    if save_model_path:
        save_model(best_model, save_model_path)
        print(f"Saved trained model to {save_model_path}")

    return best_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EEG concentration SVM from labeled CSV data")
    parser.add_argument("--csv-path", required=True, help="Path to a labeled CSV file with band features")
    parser.add_argument("--label", default="label", help="Name of the label column")
    parser.add_argument("--save-model", help="Path to save the trained sklearn model")
    args = parser.parse_args()

    main(args.csv_path, label_column=args.label, save_model_path=args.save_model)
