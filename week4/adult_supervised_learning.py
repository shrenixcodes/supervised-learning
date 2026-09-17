"""End-to-end Week 4 supervised-learning workflow for the UCI Adult dataset.

The script is deliberately self-contained: it downloads the public data, runs the
experiment, creates figures, and builds a DOCX report from the observed results.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / ".download_cache"
DATA_URLS = {
    "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
NUMERIC = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week", "capital-net", "hours-per-age",
]
CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country", "is-married", "workclass-occupation",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain features without looking at the target."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame["capital-net"] = frame["capital-gain"] - frame["capital-loss"]
        frame["hours-per-age"] = frame["hours-per-week"] / frame["age"].clip(lower=1)
        frame["is-married"] = frame["marital-status"].astype(str).str.contains("Married")
        frame["workclass-occupation"] = (
            frame["workclass"].astype(str) + " / " + frame["occupation"].astype(str)
        )
        return frame


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "week4-ml-coursework/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def load_data() -> pd.DataFrame:
    """Download and parse both official UCI train/test files."""
    train_path, test_path = CACHE / "adult.data", CACHE / "adult.test"
    download(DATA_URLS["train"], train_path)
    download(DATA_URLS["test"], test_path)
    train = pd.read_csv(train_path, names=COLUMNS, skipinitialspace=True, na_values="?")
    test = pd.read_csv(test_path, names=COLUMNS, skipinitialspace=True, na_values="?")
    test = test.iloc[1:].copy()  # adult.test has a metadata/header line
    frame = pd.concat([train, test], ignore_index=True)
    frame = frame.dropna(subset=["income"]).copy()
    for column in ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["age", "hours-per-week"])
    frame["income"] = frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    frame["income"] = (frame["income"] == ">50K").astype(int)
    return frame


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)],
        remainder="drop",
    )


def pipeline(model: Any) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", make_preprocessor()),
        ("model", model),
    ])


def metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else predicted
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
    }


def fmt_table(document: Document, frame: pd.DataFrame, digits: int = 3) -> None:
    table = document.add_table(rows=1, cols=len(frame.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    for cell, name in zip(table.rows[0].cells, frame.columns):
        cell.text = str(name)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)


def save_figures(
    data: pd.DataFrame,
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    best_model: Any,
    X_test: pd.DataFrame,
    best_name: str,
) -> dict[str, Path]:
    ARTIFACTS.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths: dict[str, Path] = {}
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x="income", hue="income", legend=False, palette="Set2")
    plt.xticks([0, 1], ["<=50K", ">50K"])
    plt.xlabel("Income class"); plt.ylabel("Records"); plt.title("Class distribution")
    plt.tight_layout(); paths["class_distribution"] = ARTIFACTS / "class_distribution.png"
    plt.savefig(paths["class_distribution"], dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    for name, score in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, score):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False-positive rate"); plt.ylabel("True-positive rate")
    plt.title("Test ROC curves"); plt.legend(); plt.tight_layout()
    paths["roc_curves"] = ARTIFACTS / "roc_curves.png"; plt.savefig(paths["roc_curves"], dpi=160); plt.close()

    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions[best_name]),
                           display_labels=["<=50K", ">50K"]).plot(cmap="Blues", values_format="d")
    plt.title(f"{best_name} confusion matrix"); plt.tight_layout()
    paths["confusion_matrix"] = ARTIFACTS / "confusion_matrix.png"
    plt.savefig(paths["confusion_matrix"], dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    for name, score in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, score)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Test precision-recall curves"); plt.legend(); plt.tight_layout()
    paths["precision_recall_curves"] = ARTIFACTS / "precision_recall_curves.png"
    plt.savefig(paths["precision_recall_curves"], dpi=160); plt.close()

    sample = X_test.sample(min(2_000, len(X_test)), random_state=RANDOM_STATE)
    sample_y = y_test.loc[sample.index]
    perm = permutation_importance(
        best_model, sample, sample_y, n_repeats=3, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )
    order = np.argsort(perm.importances_mean)[-12:]
    plt.figure(figsize=(8, 5))
    labels = X_test.columns[order]
    plt.barh(labels, perm.importances_mean[order], xerr=perm.importances_std[order], color="#4c78a8")
    plt.xlabel("Mean decrease in ROC AUC"); plt.title("Permutation importance (raw features)")
    plt.tight_layout(); paths["permutation_importance"] = ARTIFACTS / "permutation_importance.png"
    plt.savefig(paths["permutation_importance"], dpi=160); plt.close()
    return paths


def build_report(
    data: pd.DataFrame, split_sizes: tuple[int, int], cv_table: pd.DataFrame,
    test_table: pd.DataFrame, best_params: dict[str, Any], error_table: pd.DataFrame,
    paths: dict[str, Path], top_coefficients: pd.DataFrame, best_name: str,
) -> None:
    report = Document()
    styles = report.styles
    styles["Normal"].font.name = "Aptos"; styles["Normal"].font.size = Pt(10)
    title = report.add_heading("Week 4: Supervised Learning — UCI Adult Income", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = report.add_paragraph("Reproducible experiment | random seed: 42")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.add_heading("1. Objective and data", level=1)
    report.add_paragraph(
        "The objective is to classify whether annual income exceeds $50K. The UCI Adult "
        "dataset contains demographic, employment, education, and capital-gain/loss fields. "
        f"The combined dataset has {len(data):,} rows and {data.shape[1]-1} predictors; "
        f"class prevalence is {data.income.mean():.1%}."
    )
    report.add_paragraph(
        f"A stratified 80/20 holdout split produced {split_sizes[0]:,} training and "
        f"{split_sizes[1]:,} test rows. The test set was not used for model selection."
    )
    missing = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    report.add_paragraph(
        f"After parsing the two official UCI files, the analysis contains {data.shape[1] - 1} "
        f"predictors: {len(NUMERIC) - 2} original numeric fields plus engineered numeric "
        f"fields, and {len(CATEGORICAL)} categorical fields. There are {missing} remaining "
        f"missing cells after target cleaning and {duplicate_rows:,} duplicate rows; "
        "pipeline imputers handle feature missingness within each fitted training fold."
    )
    report.add_picture(str(paths["class_distribution"]), width=Inches(5.8))
    report.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.add_heading("2. Leakage-safe preparation and feature engineering", level=1)
    report.add_paragraph(
        "A custom transformer adds net capital (capital-gain minus capital-loss), "
        "hours-per-age, a married indicator, and a workclass–occupation interaction. "
        "Numeric fields use median imputation and standardization; categoricals use "
        "most-frequent imputation and one-hot encoding. Every operation is inside the "
        "pipeline, so each cross-validation fold fits preprocessing only on its training fold."
    )
    report.add_heading("3. Models and validation", level=1)
    report.add_paragraph(
        "The majority-class DummyClassifier is the baseline. Logistic regression and a "
        "class-balanced random forest are candidate models. Five-fold stratified CV was "
        "used for model comparison. Focused grid searches tuned logistic C over "
        "[0.1, 1, 10] and random-forest depth/estimators over a small, reproducible grid."
    )
    report.add_paragraph("Cross-validation results (mean across folds):")
    fmt_table(report, cv_table)
    report.add_paragraph(f"Selected {best_name} parameters: {best_params}")
    report.add_heading("4. Held-out test results", level=1)
    fmt_table(report, test_table)
    report.add_picture(str(paths["roc_curves"]), width=Inches(6.0))
    report.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.add_picture(str(paths["precision_recall_curves"]), width=Inches(6.0))
    report.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.add_picture(str(paths["confusion_matrix"]), width=Inches(4.9))
    report.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.add_heading("5. Error analysis", level=1)
    report.add_paragraph(
        f"The following table summarizes held-out {best_name.lower()} errors by observed "
        "class. False negatives (high income predicted as low) are especially important "
        "because they reduce recall; false positives represent unnecessary outreach."
    )
    fmt_table(report, error_table)
    report.add_heading("6. Interpretability", level=1)
    report.add_paragraph(
        "Permutation importance measures the decrease in test ROC AUC after shuffling each "
        "raw input feature, with the model and preprocessing kept intact. Positive logistic "
        "coefficients indicate greater odds of the >50K class after preprocessing."
    )
    report.add_picture(str(paths["permutation_importance"]), width=Inches(6.0))
    report.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.add_paragraph("Largest absolute logistic-regression coefficients:")
    fmt_table(report, top_coefficients, digits=4)
    report.add_heading("7. Limitations and next steps", level=1)
    report.add_paragraph(
        "The dataset is a historical census sample and may encode social and measurement "
        "biases; performance is not evidence of fairness or causal relationships. The "
        "single holdout is useful for coursework but confidence intervals and subgroup "
        "fairness metrics would strengthen an applied evaluation. Future work should tune "
        "the decision threshold against explicit costs, assess calibration, and report "
        "performance by sex, race, and age bands before deployment."
    )
    report.add_heading("Reproducibility", level=1)
    report.add_paragraph(
        "Run `python week4/adult_supervised_learning.py` after installing "
        "`week4/requirements.txt`. The script downloads the official UCI files and "
        "regenerates this report and all figures."
    )
    report.save(ARTIFACTS / "Week_4_Supervised_Learning_Model_Implementation_Report.docx")


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
    np.random.seed(RANDOM_STATE)
    ARTIFACTS.mkdir(exist_ok=True)
    data = load_data()
    X = data.drop(columns="income")
    y = data["income"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    baseline = pipeline(DummyClassifier(strategy="most_frequent"))
    logistic = pipeline(LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear"))
    forest = pipeline(RandomForestClassifier(
        n_estimators=180, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ))
    candidates = {
        "Majority baseline": baseline,
        "Logistic regression": logistic,
        "Random forest": forest,
    }
    cv_rows = []
    for name, model in candidates.items():
        scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            scores.append(metrics(model, X_train.iloc[valid_idx], y_train.iloc[valid_idx]))
        mean_scores = pd.DataFrame(scores).mean()
        cv_rows.append({"model": name, **mean_scores.to_dict()})

    logistic_search = GridSearchCV(
        logistic, {"model__C": [0.1, 1.0, 10.0]}, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
    )
    forest_search = GridSearchCV(
        forest, {"model__n_estimators": [120, 220], "model__max_depth": [None, 18]},
        scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
    )
    logistic_search.fit(X_train, y_train)
    forest_search.fit(X_train, y_train)
    tuned = {
        "Tuned logistic regression": logistic_search.best_estimator_,
        "Tuned random forest": forest_search.best_estimator_,
    }
    for name, model in tuned.items():
        cv_rows.append({"model": name, "cv_roc_auc": model if False else (
            logistic_search.best_score_ if "logistic" in name else forest_search.best_score_
        )})
    cv_table = pd.DataFrame(cv_rows)
    for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc"]:
        if column not in cv_table:
            cv_table[column] = np.nan
    cv_table = cv_table[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc"]]

    all_models = {**candidates, **tuned}
    test_rows, predictions, probabilities = [], {}, {}
    for name, model in all_models.items():
        if name not in tuned:
            model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)[:, 1]
        test_rows.append({"model": name, **metrics(model, X_test, y_test)})
    test_table = pd.DataFrame(test_rows)
    best_name = max(
        ("Tuned logistic regression", "Tuned random forest"),
        key=lambda name: logistic_search.best_score_ if name.endswith("logistic regression")
        else forest_search.best_score_,
    )
    best_model = tuned[best_name]
    pred = predictions[best_name]
    errors = pd.DataFrame({"actual": y_test, "predicted": pred})
    error_table = pd.DataFrame([
        {"error type": "True negatives", "count": int(((errors.actual == 0) & (errors.predicted == 0)).sum())},
        {"error type": "False positives", "count": int(((errors.actual == 0) & (errors.predicted == 1)).sum())},
        {"error type": "False negatives", "count": int(((errors.actual == 1) & (errors.predicted == 0)).sum())},
        {"error type": "True positives", "count": int(((errors.actual == 1) & (errors.predicted == 1)).sum())},
    ])
    interpretation_model = logistic_search.best_estimator_
    feature_names = interpretation_model.named_steps["preprocess"].get_feature_names_out()
    coefficients = interpretation_model.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coefficient_table["abs"] = coefficient_table.coefficient.abs()
    top_coefficients = pd.concat([
        coefficient_table.nlargest(6, "coefficient"),
        coefficient_table.nsmallest(6, "coefficient"),
    ]).drop_duplicates("feature").sort_values("coefficient", ascending=False)
    paths = save_figures(data, y_test, predictions, probabilities, best_model, X_test, best_name)
    build_report(
        data, (len(X_train), len(X_test)), cv_table, test_table,
        (logistic_search.best_params_ if best_name.endswith("logistic regression")
         else forest_search.best_params_),
        error_table, paths,
        top_coefficients[["feature", "coefficient"]].reset_index(drop=True),
        best_name,
    )
    summary = {
        "rows": len(data), "train_rows": len(X_train), "test_rows": len(X_test),
        "class_counts": data["income"].value_counts().sort_index().to_dict(),
        "best_model": best_name,
        "best_params": (logistic_search.best_params_ if best_name.endswith("logistic regression")
                        else forest_search.best_params_),
        "cv_results": cv_table.to_dict(orient="records"),
        "test_results": test_table.to_dict(orient="records"),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "best_test_metrics": test_table.loc[test_table.model == best_name].iloc[0].to_dict(),
        "artifacts": sorted(
            path.name for path in ARTIFACTS.iterdir() if path.is_file() and path.name != "results.json"
        ) + ["results.json"],
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
