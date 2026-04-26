import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Gamma and delta are ignored for the first focus model.
# OpenBCI order is delta, theta, alpha, beta, gamma, but focus detection here uses
# theta/alpha/beta plus ratios.
BASE_FEATURES = ["theta", "alpha", "beta"]
RATIO_FEATURES = ["beta_alpha_ratio", "beta_theta_ratio", "theta_alpha_ratio"]
MODEL_FEATURES = BASE_FEATURES + RATIO_FEATURES
EPSILON = 1e-9
REQUIRED_SPLIT_NAMES = ["training", "testing"]
OPTIONAL_SPLIT_NAMES = ["validation"]
FOCUSED_LABEL = "focused"
RELAXED_LABEL = "relaxed"
LABEL_ORDER = [0, 1]
LABEL_NAMES = [RELAXED_LABEL, FOCUSED_LABEL]
DEFAULT_SMOOTH_WINDOW = 5

# New data should use only the canonical labels above.
# Older task-specific labels are accepted only so existing CSVs remain usable.
FOCUSED_ALIASES = {"1", "true", FOCUSED_LABEL, "focused_reading", "focused_math", "reading", "math"}
RELAXED_ALIASES = {"0", "false", RELAXED_LABEL, "unfocused", "not_focused", "relax", "rest"}


def normalize_label_text(value):
    text = str(value).strip().lower()
    if text in FOCUSED_ALIASES:
        return FOCUSED_LABEL
    if text in RELAXED_ALIASES:
        return RELAXED_LABEL
    raise ValueError(
        f"Unknown label: {value}. Use canonical labels '{FOCUSED_LABEL}' or '{RELAXED_LABEL}'."
    )


def encode_label(value):
    return 1 if normalize_label_text(value) == FOCUSED_LABEL else 0


def print_progress(message):
    print(f"[progress] {message}")


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


def load_feature_file(path):
    """Load a CSV-formatted feature file, even if it has no .csv extension."""
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def looks_like_feature_file(path, label_column="label"):
    """Return True when a file has the required CSV header for model training."""
    path = Path(path)
    if not path.is_file() or path.name.startswith("."):
        return False
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    normalized_columns = {str(col).strip() for col in header}
    required = {label_column, "theta", "alpha", "beta"}
    return required.issubset(normalized_columns)


def load_band_features_from_dataframe(df, label_column="label"):
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    y = df[label_column].apply(encode_label).astype(int)
    X = add_ratio_features(df)
    X = X[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def load_band_features(csv_path, label_column="label"):
    df = load_feature_file(csv_path)
    return load_band_features_from_dataframe(df, label_column=label_column)


def find_feature_files(folder, label_column="label"):
    """Find all CSV-formatted feature files in a split folder.

    The training interface may produce files named like `Sampo0-features` without
    a `.csv` suffix. This function checks file contents instead of extension.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset split folder does not exist: {folder}")

    feature_files = sorted(
        path for path in folder.iterdir()
        if looks_like_feature_file(path, label_column=label_column)
    )
    if not feature_files:
        raise FileNotFoundError(
            f"No CSV-formatted feature files found in dataset split folder: {folder}. "
            "Expected files with columns: label, theta, alpha, beta. The file does not need a .csv extension."
        )
    return feature_files


def split_folder_has_feature_files(folder, label_column="label"):
    folder = Path(folder)
    return folder.exists() and any(looks_like_feature_file(path, label_column=label_column) for path in folder.iterdir())


def load_split_folder(folder, label_column="label"):
    frames = []
    feature_files = find_feature_files(folder, label_column=label_column)
    for feature_file in feature_files:
        df = load_feature_file(feature_file)
        df["source_file"] = feature_file.name
        frames.append(df)
    split_df = pd.concat(frames, ignore_index=True)
    X, y = load_band_features_from_dataframe(split_df, label_column=label_column)
    return X, y, feature_files


def load_dataset_folder(dataset_dir, label_column="label"):
    dataset_dir = Path(dataset_dir)
    print_progress(f"Loading dataset directory: {dataset_dir}")
    splits = {}

    for split_name in REQUIRED_SPLIT_NAMES:
        split_path = dataset_dir / split_name
        X, y, files = load_split_folder(split_path, label_column=label_column)
        splits[split_name] = {"X": X, "y": y, "files": files}

    for split_name in OPTIONAL_SPLIT_NAMES:
        split_path = dataset_dir / split_name
        if split_folder_has_feature_files(split_path, label_column=label_column):
            X, y, files = load_split_folder(split_path, label_column=label_column)
            splits[split_name] = {"X": X, "y": y, "files": files}
        else:
            print(f"Optional split '{split_name}' not found or empty; skipping.")

    return splits


def print_label_distribution(name, y):
    counts = pd.Series(y).value_counts().sort_index()
    readable = {f"{RELAXED_LABEL}(0)": int(counts.get(0, 0)), f"{FOCUSED_LABEL}(1)": int(counts.get(1, 0))}
    print(f"{name} label distribution: {readable}")


def require_two_classes(y, split_name):
    unique = pd.Series(y).unique()
    if len(unique) != 2:
        raise ValueError(
            f"{split_name} split has only {len(unique)} class(es): {sorted(unique)}. "
            f"Each split should contain both labels: {FOCUSED_LABEL} and {RELAXED_LABEL}."
        )


def require_min_samples_per_class(y, split_name, min_samples=2):
    counts = pd.Series(y).value_counts()
    if (counts < min_samples).any():
        raise ValueError(
            f"{split_name} split must contain at least {min_samples} samples for each label. "
            f"Label counts: {counts.to_dict()}"
        )


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


def smooth_windows(X, window_size=DEFAULT_SMOOTH_WINDOW, min_periods=1):
    if window_size is None or window_size <= 1:
        return X
    return X.rolling(window=window_size, min_periods=min_periods, center=True).mean().bfill().ffill()


def prepare_features(X, z_thresh=3.5, smooth_window=DEFAULT_SMOOTH_WINDOW, fill_method="median"):
    Xp = X.copy()
    Xp = Xp.apply(pd.to_numeric, errors="coerce")
    Xp = Xp.replace([np.inf, -np.inf], np.nan)
    Xp = Xp.fillna(Xp.median(numeric_only=True))
    Xp = replace_outliers(Xp, z_thresh=z_thresh, fill_method=fill_method)
    Xp = smooth_windows(Xp, window_size=smooth_window)
    return Xp


def make_calibrated_classifier(base_classifier, cv):
    """Create CalibratedClassifierCV across sklearn versions.

    sklearn renamed base_estimator to estimator in newer versions. This helper
    keeps the script usable on both old and new sklearn installs.
    """
    try:
        return CalibratedClassifierCV(estimator=base_classifier, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_classifier, method="sigmoid", cv=cv)


def build_pipeline(rbf_gamma=1.0, rbf_components=300, svm_c=1.0, calibration_cv=3, random_state=42):
    """Build a fast non-linear SVM pipeline.

    Full RBF SVC training is slow on large EEG datasets. This keeps non-linearity
    by approximating the RBF kernel with RBFSampler, then trains a LinearSVC.
    The final calibrated wrapper provides predict_proba() for confidence scores.
    """
    linear_svm = LinearSVC(
        C=svm_c,
        class_weight="balanced",
        max_iter=10000,
        random_state=random_state,
    )
    calibrated_svm = make_calibrated_classifier(linear_svm, cv=calibration_cv)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rbf_features", RBFSampler(gamma=rbf_gamma, n_components=rbf_components, random_state=random_state)),
            ("svc", calibrated_svm),
        ]
    )


def train_fast_svm(
    X,
    y,
    rbf_gamma=1.0,
    rbf_components=300,
    svm_c=1.0,
    calibration_cv=3,
    random_state=42,
):
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError(
            f"Need at least 2 samples per class for calibrated SVM training. Label counts: {class_counts.to_dict()}"
        )
    calibration_cv = max(2, min(calibration_cv, min_class_count))
    print_progress(
        "Training fast SVM: "
        f"samples={len(X)}, class_counts={class_counts.to_dict()}, "
        f"rbf_gamma={rbf_gamma}, rbf_components={rbf_components}, C={svm_c}, calibration_cv={calibration_cv}"
    )
    model = build_pipeline(
        rbf_gamma=rbf_gamma,
        rbf_components=rbf_components,
        svm_c=svm_c,
        calibration_cv=calibration_cv,
        random_state=random_state,
    )
    model.fit(X, y)
    print_progress("Fast SVM training complete")
    return model


def confusion_matrix_dataframe(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in LABEL_NAMES],
        columns=[f"predicted_{name}" for name in LABEL_NAMES],
    )


def save_confusion_matrix_csv(cm_df, split_name, output_dir):
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_{split_name}.csv"
    cm_df.to_csv(path)
    print(f"Saved {split_name} confusion matrix to {path}")


def print_confusion_summary(split_name, cm_df):
    print(f"\n{split_name} confusion matrix table:")
    print(cm_df.to_string())
    tn = int(cm_df.loc[f"actual_{RELAXED_LABEL}", f"predicted_{RELAXED_LABEL}"])
    fp = int(cm_df.loc[f"actual_{RELAXED_LABEL}", f"predicted_{FOCUSED_LABEL}"])
    fn = int(cm_df.loc[f"actual_{FOCUSED_LABEL}", f"predicted_{RELAXED_LABEL}"])
    tp = int(cm_df.loc[f"actual_{FOCUSED_LABEL}", f"predicted_{FOCUSED_LABEL}"])
    print(f"True relaxed: {tn} | False focused: {fp} | False relaxed: {fn} | True focused: {tp}")


def evaluate_model(model, X_test, y_test, split_name="test", confusion_output_dir=None):
    y_pred = model.predict(X_test)
    print(
        f"\n{split_name} classification report:\n",
        classification_report(
            y_test,
            y_pred,
            digits=4,
            labels=LABEL_ORDER,
            target_names=LABEL_NAMES,
            zero_division=0,
        ),
    )
    cm_df = confusion_matrix_dataframe(y_test, y_pred)
    print_confusion_summary(split_name, cm_df)
    save_confusion_matrix_csv(cm_df, split_name, confusion_output_dir)
    return cm_df


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def predict_focus(model, X, threshold=0.5, label_map=None):
    if label_map is None:
        label_map = {1: FOCUSED_LABEL, 0: RELAXED_LABEL}

    X = add_ratio_features(X)
    X_prepared = prepare_features(X[MODEL_FEATURES], z_thresh=3.5, smooth_window=DEFAULT_SMOOTH_WINDOW, fill_method="median")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support probability estimates")

    proba = model.predict_proba(X_prepared)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Expected trained labels to contain 1")

    positive_probs = proba[:, classes.index(1)]
    labels = [label_map[1] if p >= threshold else label_map[0] for p in positive_probs]
    return pd.DataFrame({"confidence": positive_probs, "focus": labels}, index=X.index)


def train_from_single_csv(
    csv_path,
    label_column="label",
    save_model_path=None,
    confusion_output_dir=None,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    rbf_gamma=1.0,
    rbf_components=300,
    svm_c=1.0,
    calibration_cv=3,
):
    print_progress(f"Loading single CSV feature file: {csv_path}")
    X, y = load_band_features(csv_path, label_column=label_column)
    print_label_distribution("full csv", y)
    require_two_classes(y, "full csv")
    require_min_samples_per_class(y, "full csv", min_samples=2)

    print_progress("Preparing features and splitting data")
    X_prepared = prepare_features(X, z_thresh=3.5, smooth_window=smooth_window, fill_method="median")
    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    best_model = train_fast_svm(
        X_train,
        y_train,
        rbf_gamma=rbf_gamma,
        rbf_components=rbf_components,
        svm_c=svm_c,
        calibration_cv=calibration_cv,
    )
    print_progress("Evaluating random holdout data")
    evaluate_model(best_model, X_test, y_test, split_name="random_holdout", confusion_output_dir=confusion_output_dir)

    if save_model_path:
        save_model(best_model, save_model_path)
        print(f"Saved trained model to {save_model_path}")
    return best_model


def train_from_dataset_folder(
    dataset_dir,
    label_column="label",
    save_model_path=None,
    confusion_output_dir=None,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    rbf_gamma=1.0,
    rbf_components=300,
    svm_c=1.0,
    calibration_cv=3,
):
    print_progress(f"Training from dataset folder: {dataset_dir}")
    splits = load_dataset_folder(dataset_dir, label_column=label_column)

    for split_name, split in splits.items():
        loaded_files = ", ".join(str(path.name) for path in split["files"])
        print(f"Loaded {len(split['files'])} feature file(s) from {Path(dataset_dir) / split_name}: {loaded_files}")
        print_label_distribution(split_name, split["y"])
        require_two_classes(split["y"], split_name)
        require_min_samples_per_class(split["y"], split_name, min_samples=2)

    X_train = prepare_features(splits["training"]["X"], z_thresh=3.5, smooth_window=smooth_window, fill_method="median")
    y_train = splits["training"]["y"]
    X_test = prepare_features(splits["testing"]["X"], z_thresh=3.5, smooth_window=smooth_window, fill_method="median")
    y_test = splits["testing"]["y"]

    best_model = train_fast_svm(
        X_train,
        y_train,
        rbf_gamma=rbf_gamma,
        rbf_components=rbf_components,
        svm_c=svm_c,
        calibration_cv=calibration_cv,
    )

    print_progress("Evaluating testing split")
    evaluate_model(best_model, X_test, y_test, split_name="testing", confusion_output_dir=confusion_output_dir)

    if "validation" in splits:
        X_val = prepare_features(splits["validation"]["X"], z_thresh=3.5, smooth_window=smooth_window, fill_method="median")
        y_val = splits["validation"]["y"]
        print_progress("Evaluating validation split")
        evaluate_model(best_model, X_val, y_val, split_name="validation", confusion_output_dir=confusion_output_dir)

    if save_model_path:
        save_model(best_model, save_model_path)
        print(f"Saved trained model to {save_model_path}")
    return best_model


def main(
    csv_path=None,
    dataset_dir=None,
    label_column="label",
    save_model_path=None,
    confusion_output_dir=None,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    rbf_gamma=1.0,
    rbf_components=300,
    svm_c=1.0,
    calibration_cv=3,
):
    if dataset_dir:
        print_progress(f"Selected dataset folder mode: {dataset_dir}")
        return train_from_dataset_folder(
            dataset_dir,
            label_column=label_column,
            save_model_path=save_model_path,
            confusion_output_dir=confusion_output_dir,
            smooth_window=smooth_window,
            rbf_gamma=rbf_gamma,
            rbf_components=rbf_components,
            svm_c=svm_c,
            calibration_cv=calibration_cv,
        )
    if csv_path:
        return train_from_single_csv(
            csv_path,
            label_column=label_column,
            save_model_path=save_model_path,
            confusion_output_dir=confusion_output_dir,
            smooth_window=smooth_window,
            rbf_gamma=rbf_gamma,
            rbf_components=rbf_components,
            svm_c=svm_c,
            calibration_cv=calibration_cv,
        )
    raise ValueError("Provide either --csv-path or --dataset-dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG focus SVM from labeled OpenBCI band features")
    parser.add_argument("--csv-path", help="Path to one feature file, with or without .csv extension")
    parser.add_argument("--dataset-dir", help="Folder containing required training/ and testing/ feature folders, plus optional validation/")
    parser.add_argument("--label", default="label", help="Name of the label column")
    parser.add_argument("--save-model", help="Path to save the trained sklearn model")
    parser.add_argument("--confusion-output-dir", help="Optional folder to save confusion_matrix_<split>.csv files")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Smoothing window size. Default: 5")
    parser.add_argument("--rbf-gamma", type=float, default=1.0, help="RBFSampler gamma. Default: 1.0")
    parser.add_argument("--rbf-components", type=int, default=300, help="Number of RBF approximation components. Default: 300")
    parser.add_argument("--svm-c", type=float, default=1.0, help="Linear SVM C value. Default: 1.0")
    parser.add_argument("--calibration-cv", type=int, default=3, help="Probability calibration folds. Default: 3")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        dataset_dir=args.dataset_dir,
        label_column=args.label,
        save_model_path=args.save_model,
        confusion_output_dir=args.confusion_output_dir,
        smooth_window=args.smooth_window,
        rbf_gamma=args.rbf_gamma,
        rbf_components=args.rbf_components,
        svm_c=args.svm_c,
        calibration_cv=args.calibration_cv,
    )
