import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


BASE_FEATURES = ["delta", "theta", "alpha", "beta"]
RATIO_FEATURES = ["beta_alpha_ratio", "beta_theta_ratio", "theta_alpha_ratio"]
MODEL_FEATURES = BASE_FEATURES + RATIO_FEATURES
EPSILON = 1e-9


def encode_label(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "focused", "focused_reading", "focused_math", "reading", "math"}:
        return 1
    if text in {"0", "false", "relaxed", "unfocused", "not_focused", "relax", "rest"}:
        return 0
    raise ValueError(f"Unknown label: {value}")


def add_ratio_features(df):
    X = df.copy()
    for col in BASE_FEATURES:
        if col not in X.columns:
            raise ValueError(f"Missing required feature column: {col}")
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X["beta_alpha_ratio"] = X["beta"] / (X["alpha"] + EPSILON)
    X["beta_theta_ratio"] = X["beta"] / (X["theta"] + EPSILON)
    X["theta_alpha_ratio"] = X["theta"] / (X["alpha"] + EPSILON)
    return X


def load_band_features(csv_path, label_column="label"):
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    y = df[label_column].apply(encode_label).astype(int)
    X = add_ratio_features(df)
    X = X[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def replace_outliers(X, z_thresh=3.0, fill_method="median"):
    Xr = X.copy()
    med = Xr.median(numeric_only=True)
    mad = (Xr - med).abs().median(numeric_only=True).replace(0, np.finfo(float).eps)
    z = ((Xr - med).abs() / mad).replace([np.inf, -np.inf], np.nan).fillna(0)
    mask = z > z_thresh

    if fill_method == "median":
        for column in Xr.columns:
            Xr.loc[mask[column], column] = med[column]
    elif fill_method == "clip":
        lower = med - z_thresh * mad
        upper = med + z_thresh * mad
        Xr = Xr.clip(lower=lower, upper=upper, axis=1)
    else:
        raise ValueError("fill_method must be 'median' or 'clip'")
    return Xr


def smooth_windows(X, window_size=3, min_periods=1):
    if window_size is None or window_size <= 1:
        return X
    return X.rolling(window=window_size, min_periods=min_periods, center=True).mean().bfill().ffill()


def prepare_features(X, z_thresh=3.5, smooth_window=3, fill_method="median"):
    Xp = X.copy()
    Xp = Xp.apply(pd.to_numeric, errors="coerce")
    Xp = Xp.replace([np.inf, -np.inf], np.nan)
    Xp = Xp.fillna(Xp.median(numeric_only=True))
    Xp = replace_outliers(Xp, z_thresh=z_thresh, fill_method=fill_method)
    Xp = smooth_windows(Xp, window_size=smooth_window)
    return Xp


def build_pipeline(use_pca=True, pca_variance=0.95, scaler="standard", probability=True):
    steps = []
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    else:
        raise ValueError("scaler must be 'standard' or 'robust'")

    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))

    steps.append(("svc", SVC(kernel="rbf", probability=probability, class_weight="balanced")))
    return Pipeline(steps)


def tune_and_train(X, y, use_pca=True, scaler="standard", probability=True):
    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    pipe = build_pipeline(use_pca=use_pca, scaler=scaler, probability=probability)
    param_grid = {"svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", "auto"]}
    search = GridSearchCV(
        pipe,
        param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    return search


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def predict_focus(model, X, threshold=0.5, label_map=None):
    if label_map is None:
        label_map = {1: "focused", 0: "unfocused"}

    X = add_ratio_features(X)
    X_prepared = prepare_features(X[MODEL_FEATURES], z_thresh=3.5, smooth_window=3, fill_method="median")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support probability estimates")

    proba = model.predict_proba(X_prepared)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Expected trained labels to contain 1")

    positive_probs = proba[:, classes.index(1)]
    labels = [label_map[1] if p >= threshold else label_map[0] for p in positive_probs]
    return pd.DataFrame({"confidence": positive_probs, "focus": labels}, index=X.index)


def main(csv_path, label_column="label", save_model_path=None):
    X, y = load_band_features(csv_path, label_column=label_column)
    X_prepared = prepare_features(X, z_thresh=3.5, smooth_window=3, fill_method="median")

    if y.nunique() != 2:
        raise ValueError("Training requires both focused and relaxed/unfocused labels")

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
    parser = argparse.ArgumentParser(description="Train EEG focus SVM from labeled OpenBCI band features")
    parser.add_argument("--csv-path", required=True, help="Path to <subject>-features.csv or raw collector CSV")
    parser.add_argument("--label", default="label", help="Name of the label column")
    parser.add_argument("--save-model", help="Path to save the trained sklearn model")
    args = parser.parse_args()
    main(args.csv_path, label_column=args.label, save_model_path=args.save_model)
