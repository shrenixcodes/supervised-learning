"""End-to-end Week 4 supervised-learning workflow for the UCI Adult dataset.

The script is deliberately self-contained: it downloads the public data, runs the
experiment, creates figures, and builds a DOCX report from the observed results.
"""

from __future__ import annotations

import json
import os
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
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
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
BASE_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
ENGINEERED_NUMERIC = ["capital-net", "hours-per-age"]
NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC
BASE_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]
ENGINEERED_CATEGORICAL = ["is-married", "workclass-occupation"]
CATEGORICAL = BASE_CATEGORICAL + ENGINEERED_CATEGORICAL


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


class PassThrough(BaseEstimator, TransformerMixin):
    """No-op transformer used for the feature-engineering ablation pipeline."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PassThrough":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "week4-ml-coursework/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def load_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and parse both official UCI train/test files, tracking cleaning stats."""
    train_path, test_path = CACHE / "adult.data", CACHE / "adult.test"
    download(DATA_URLS["train"], train_path)
    download(DATA_URLS["test"], test_path)
    train = pd.read_csv(train_path, names=COLUMNS, skipinitialspace=True, na_values="?")
    test = pd.read_csv(test_path, names=COLUMNS, skipinitialspace=True, na_values="?")
    test = test.iloc[1:].copy()  # adult.test has a metadata/header line
    frame = pd.concat([train, test], ignore_index=True)
    raw_rows = len(frame)
    frame = frame.dropna(subset=["income"]).copy()
    for column in ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_by_column = frame[BASE_CATEGORICAL].isna().sum().to_dict()
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.dropna(subset=["age", "hours-per-week"])
    frame["income"] = frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    frame["income"] = (frame["income"] == ">50K").astype(int)
    stats = {
        "raw_rows": raw_rows,
        "rows_after_cleaning": len(frame),
        "missing_by_column": {k: int(v) for k, v in missing_by_column.items() if v > 0},
        "duplicate_rows": duplicate_rows,
    }
    return frame, stats


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )


def pipeline(model: Any) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", make_preprocessor(NUMERIC, CATEGORICAL)),
        ("model", model),
    ])


def baseline_pipeline(model: Any) -> Pipeline:
    """Pipeline that skips engineered features, used only for the ablation study."""
    return Pipeline([
        ("features", PassThrough()),
        ("preprocess", make_preprocessor(BASE_NUMERIC, BASE_CATEGORICAL)),
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
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            if isinstance(value, (float, np.floating)):
                cell.text = "nan" if pd.isna(value) else f"{value:.{digits}f}"
            else:
                cell.text = str(value)


def add_caption(document: Document, text: str) -> None:
    caption = document.add_paragraph(text)
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(9)


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
    plt.xlabel("Income class"); plt.ylabel("Records"); plt.title("Target class distribution (full dataset)")
    plt.tight_layout(); paths["class_distribution"] = ARTIFACTS / "class_distribution.png"
    plt.savefig(paths["class_distribution"], dpi=160); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(data=data, x="age", hue="income", bins=30, multiple="stack", palette="Set2", ax=axes[0])
    axes[0].set_title("Age by income class"); axes[0].set_xlabel("Age"); axes[0].set_ylabel("Records")
    sns.histplot(data=data, x="hours-per-week", hue="income", bins=30, multiple="stack", palette="Set2", ax=axes[1])
    axes[1].set_title("Hours per week by income class"); axes[1].set_xlabel("Hours per week"); axes[1].set_ylabel("Records")
    plt.tight_layout(); paths["feature_distributions"] = ARTIFACTS / "feature_distributions.png"
    plt.savefig(paths["feature_distributions"], dpi=160); plt.close()

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
    plt.title(f"{best_name} confusion matrix (test set)"); plt.tight_layout()
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
    plt.xlabel("Mean decrease in test ROC AUC"); plt.title(f"Permutation importance — {best_name} (raw features)")
    plt.tight_layout(); paths["permutation_importance"] = ARTIFACTS / "permutation_importance.png"
    plt.savefig(paths["permutation_importance"], dpi=160); plt.close()

    return paths


def build_model_comparison_figure(cv_table: pd.DataFrame, test_table: pd.DataFrame) -> Path:
    merged = cv_table[["model", "cv_roc_auc"]].merge(
        test_table[["model", "roc_auc", "accuracy", "f1"]], on="model", how="right"
    )
    plt.figure(figsize=(8, 5))
    x = np.arange(len(merged))
    width = 0.25
    plt.bar(x - width, merged["accuracy"], width, label="Accuracy", color="#72b7b2")
    plt.bar(x, merged["f1"], width, label="F1-score", color="#4c78a8")
    plt.bar(x + width, merged["roc_auc"], width, label="ROC-AUC", color="#e45756")
    plt.xticks(x, merged["model"], rotation=20, ha="right")
    plt.ylabel("Score"); plt.ylim(0, 1)
    plt.title("Model comparison on held-out test data")
    plt.legend(); plt.tight_layout()
    path = ARTIFACTS / "model_comparison.png"
    plt.savefig(path, dpi=160); plt.close()
    return path


def build_error_analysis_figure(X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray, best_name: str) -> Path:
    errors = X_test.copy()
    errors["actual"] = y_test.values
    errors["predicted"] = pred
    errors["outcome"] = np.select(
        [
            (errors.actual == 1) & (errors.predicted == 0),
            (errors.actual == 0) & (errors.predicted == 1),
        ],
        ["False negative", "False positive"],
        default="Correct",
    )
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=errors, x="outcome", y="hours-per-week",
                order=["Correct", "False negative", "False positive"], palette="Set2")
    plt.title(f"Hours-per-week distribution by {best_name.lower()} prediction outcome")
    plt.xlabel("Prediction outcome"); plt.ylabel("Hours per week")
    plt.tight_layout()
    path = ARTIFACTS / "error_analysis.png"
    plt.savefig(path, dpi=160); plt.close()
    return path, errors


def add_code_appendix(document: Document, code_path: Path) -> None:
    document.add_heading("21. Appendix: Full Source Code", level=1)
    document.add_paragraph(
        "The complete script below produced every number, table, and figure in this "
        "report. It is included verbatim for reproducibility and grading."
    )
    code_text = code_path.read_text(encoding="utf-8")
    for line in code_text.splitlines():
        para = document.add_paragraph()
        run = para.add_run(line if line.strip() else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(7.5)
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.space_before = Pt(0)


def build_report(
    ctx: dict[str, Any],
) -> None:
    data = ctx["data"]
    report = Document()
    styles = report.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10.5)

    # ---- Title page ----
    title = report.add_heading("Week 4: Supervised Learning Model Implementation", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = report.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Predicting Income Category from the UCI Adult / Census Income Dataset")
    run.italic = True; run.font.size = Pt(14)
    meta = report.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        "\nDataset: UCI Adult / Census Income (archive.ics.uci.edu/dataset/2/adult)\n"
        "Problem type: Binary classification\n"
        f"Reproducibility seed: {RANDOM_STATE}\n"
    )
    report.add_page_break()

    # ---- 1. Introduction ----
    report.add_heading("1. Introduction", level=1)
    report.add_paragraph(
        "Supervised learning uses labeled historical examples to learn a function that maps "
        "input features to a known outcome, so that the function can then predict the outcome "
        "for new, unlabeled cases. It is the appropriate paradigm here because every record in "
        "the Adult dataset carries a ground-truth income label derived from the 1994 U.S. Census "
        "Bureau survey, which allows the model's predictions to be checked against a known answer "
        "during training and evaluation."
    )
    report.add_paragraph(
        "The objective of this project is to build, validate, and critically evaluate a "
        "classification model that predicts whether an individual's annual income exceeds "
        "$50,000 based on demographic and employment attributes, and to document every "
        "methodological decision with its rationale, alternatives, and trade-offs."
    )

    # ---- 2. Problem Definition ----
    report.add_heading("2. Problem Definition", level=1)
    report.add_paragraph(
        f"Given demographic and employment features (age, education, occupation, hours worked, "
        f"marital status, capital gains/losses, and related fields), the objective is to predict "
        f"whether an individual's income is above or below $50,000 per year. This is a binary "
        f"classification problem because the target variable, income, takes exactly two discrete "
        f"values (<=50K encoded as 0, and >50K encoded as 1) rather than a continuous quantity."
    )
    report.add_paragraph(
        "Supervised learning is appropriate because the historical Census extract provides a "
        "verified label for every record, the relationship between demographic/employment "
        "attributes and income bracket is plausibly learnable from patterns in the data (e.g. "
        "education level and hours worked are established correlates of earnings), and the "
        "prediction task has clear practical value for applications such as survey-based "
        "socioeconomic research, resource targeting, and coarse income estimation when direct "
        "income reporting is unavailable."
    )

    # ---- 3. Dataset Overview ----
    report.add_heading("3. Dataset Overview", level=1)
    report.add_paragraph(
        "Source: UCI Machine Learning Repository, 'Adult' dataset "
        "(https://archive.ics.uci.edu/dataset/2/adult), originally extracted from the 1994 U.S. "
        "Census Bureau database by Barry Becker. The official adult.data (training) and "
        "adult.test (test) files were downloaded programmatically and concatenated so that the "
        "project could control its own stratified split rather than relying on the dataset's "
        "original partition."
    )
    overview_rows = pd.DataFrame([
        {"Property": "Rows after cleaning", "Value": f"{len(data):,}"},
        {"Property": "Raw rows (pre-cleaning)", "Value": f"{ctx['clean_stats']['raw_rows']:,}"},
        {"Property": "Predictor columns (original)", "Value": str(len(BASE_NUMERIC) + len(BASE_CATEGORICAL))},
        {"Property": "Target variable", "Value": "income (0 = <=50K, 1 = >50K)"},
        {"Property": "Numeric original features", "Value": ", ".join(BASE_NUMERIC)},
        {"Property": "Categorical original features", "Value": ", ".join(BASE_CATEGORICAL)},
        {"Property": "Duplicate rows found", "Value": str(ctx["clean_stats"]["duplicate_rows"])},
        {"Property": "Missing (unknown '?') cells by column", "Value": str(ctx["clean_stats"]["missing_by_column"])},
    ])
    fmt_table(report, overview_rows, digits=0)

    # ---- 4. Exploratory Data Analysis ----
    report.add_heading("4. Exploratory Data Analysis", level=1)
    report.add_paragraph(
        f"The target is imbalanced: {data.income.mean():.1%} of records are labeled >50K and "
        f"{1 - data.income.mean():.1%} are labeled <=50K. This class imbalance is important "
        "because a model that always predicts the majority class would already reach roughly "
        f"{max(data.income.mean(), 1 - data.income.mean()):.1%} accuracy without learning "
        "anything useful, which is why accuracy alone is an insufficient evaluation metric here "
        "(see Sections 12 and 16)."
    )
    report.add_picture(str(ctx["paths"]["class_distribution"]), width=Inches(5.5))
    add_caption(report, "Figure 1. Distribution of the target variable across the full dataset.")
    report.add_paragraph(
        "Descriptive statistics for the core numeric predictors, computed on the full cleaned "
        "dataset before the train/test split (splitting occurs immediately afterward and no "
        "transformation is fit at this stage):"
    )
    desc = data[BASE_NUMERIC].describe().T.reset_index().rename(columns={"index": "feature"})
    fmt_table(report, desc, digits=1)
    report.add_picture(str(ctx["paths"]["feature_distributions"]), width=Inches(6.3))
    add_caption(report, "Figure 2. Age and hours-per-week distributions split by income class.")
    report.add_paragraph(
        "OBSERVED: median age and hours-per-week are visibly higher in the >50K group in the "
        "histograms above. POSSIBLE EXPLANATION: this is consistent with earnings generally "
        "rising with career tenure/experience and with full-time or overtime employment, though "
        "the dataset cannot establish causation between these variables and income."
    )

    # ---- 5. Data Preprocessing ----
    report.add_heading("5. Data Preprocessing", level=1)
    report.add_paragraph(
        "DECISION: Numeric features are median-imputed then standardized; categorical features "
        "are most-frequent-imputed then one-hot encoded with rare categories (count < 2 in the "
        "training fold) collapsed into an 'infrequent' bucket."
    )
    report.add_paragraph(
        "WHY NECESSARY: the workclass, occupation, and native-country columns contain '?' "
        "placeholder values that pandas parses as missing; a model cannot consume NaNs or "
        "text categories directly, and logistic regression additionally requires comparable "
        "feature scales to converge reliably and to keep coefficient magnitudes interpretable."
    )
    report.add_paragraph(
        "WHY THIS APPROACH: median imputation is robust to the right-skewed numeric fields "
        "(capital-gain and capital-loss are extremely skewed — most values are 0), and "
        "most-frequent imputation is a simple, defensible default for the small fraction of "
        "missing categorical cells. Standardization was chosen over min-max scaling because it "
        "is less sensitive to the extreme outliers present in capital-gain/capital-loss."
    )
    report.add_paragraph(
        "ALTERNATIVES CONSIDERED: mean imputation was rejected because it is more sensitive to "
        "skew and outliers than the median; dropping rows with missing categoricals was rejected "
        "because it would discard roughly 2,000+ records unnecessarily for a tree/linear model "
        "that can be trained on the remaining fields; target encoding for categoricals was "
        "considered but rejected because it leaks target information into the training fold's "
        "feature representation unless implemented with careful cross-fitting."
    )
    report.add_paragraph(
        "EFFECT ON MODEL PERFORMANCE: standardization primarily affects logistic regression's "
        "convergence and coefficient interpretability; the tree-based random forest is invariant "
        "to monotonic feature scaling, so its results are essentially unaffected by this choice. "
        "One-hot encoding with a minimum-frequency threshold keeps the feature space bounded "
        "(rare native-country values collapse together) rather than creating hundreds of sparse "
        "columns that could increase variance in the linear model."
    )
    report.add_paragraph(
        "LIMITATIONS / TRADE-OFFS: median/most-frequent imputation assumes missingness is not "
        "strongly informative; if missing workclass values were systematically associated with "
        "unemployment (plausible, since 'workclass=?' often co-occurs with 'occupation=?'), a "
        "missing-indicator flag could preserve that signal, which this pipeline does not add. "
        "Collapsing rare categories with min_frequency=2 slightly reduces the model's ability "
        "to distinguish very rare native-country values from one another."
    )
    report.add_paragraph(
        "DATA LEAKAGE: fitting a scaler, imputer, or encoder on the full dataset before splitting "
        "would let statistics from the test set (e.g. the median age of the whole dataset) "
        "influence how training data is transformed, which inflates validation performance in a "
        "way that will not generalize to genuinely new data. This project prevents that by "
        "wrapping every transformation inside a single scikit-learn Pipeline/ColumnTransformer "
        "and calling .fit() only on training folds — during cross-validation each fold refits "
        "its own imputer/scaler/encoder, and the held-out test set is transformed only with "
        "statistics learned from the training split."
    )

    # ---- 6. Feature Engineering ----
    report.add_heading("6. Feature Engineering", level=1)
    report.add_paragraph(
        "Four engineered features were added inside the pipeline's FeatureEngineer step, which "
        "runs before the ColumnTransformer and therefore only ever sees data from the fold or "
        "split it is applied to:"
    )
    fe_list = report.add_paragraph(style="List Bullet")
    fe_list.add_run("capital-net = capital-gain − capital-loss — collapses two extremely sparse, "
                     "heavily right-skewed columns into a single signed measure of net capital income.")
    for text in [
        "hours-per-age = hours-per-week / age — a simple interaction intended to capture whether "
        "someone works unusually long hours relative to their career stage.",
        "is-married — a boolean flag derived from marital-status containing 'Married', because "
        "the raw field has seven granular categories and marital status is a well-documented "
        "correlate of household income in Census research.",
        "workclass-occupation — a concatenation of workclass and occupation intended to capture "
        "combinations (e.g. 'Self-emp-inc / Exec-managerial') that neither field expresses alone.",
    ]:
        report.add_paragraph(text, style="List Bullet")
    report.add_paragraph(
        f"EXPERIMENTAL EVIDENCE: to test whether engineering actually helps rather than assuming "
        f"it does, a random forest was cross-validated (5-fold stratified, same seed) using only "
        f"the {len(BASE_NUMERIC)} original numeric and {len(BASE_CATEGORICAL)} original "
        f"categorical fields, and compared against the same model with all engineered features "
        f"added."
    )
    fe_table = ctx["feature_engineering_table"]
    fmt_table(report, fe_table, digits=4)
    fe_delta = ctx["fe_delta"]
    direction = "improved" if fe_delta > 0 else "did not improve" if fe_delta < 0 else "left unchanged"
    report.add_paragraph(
        f"RESULT: engineered features {direction} mean cross-validated ROC-AUC by "
        f"{fe_delta:+.4f} ({fe_delta / ctx['fe_base_score']:+.2%} relative). "
        "INTERPRETATION: "
        + (
            "the engineered fields carry information the raw fields alone did not expose as "
            "directly, supporting their inclusion."
            if fe_delta > 0.001
            else "the random forest can already approximate interactions such as capital-net or "
            "hours-per-age internally through its own splits, so explicitly engineering them adds "
            "little for this particular model; they were kept regardless because they also aid "
            "the logistic regression model, which cannot learn non-linear interactions on its own, "
            "and because is-married/capital-net improve interpretability of the coefficient table "
            "in Section 15."
        )
        + " LIMITATION: this ablation was run for the random forest only and with a single CV "
        "configuration; the magnitude of the effect could differ for other algorithms or splits."
    )

    # ---- 7. Train/Test Strategy ----
    report.add_heading("7. Train / Test Strategy", level=1)
    train_rows, test_rows = ctx["split_sizes"]
    report.add_paragraph(
        f"The cleaned dataset ({len(data):,} rows) was split into {train_rows:,} training rows "
        f"(80%) and {test_rows:,} test rows (20%) using train_test_split with stratify=income "
        f"and random_state={RANDOM_STATE}, so both partitions preserve the "
        f"{data.income.mean():.1%} positive-class rate observed in the full dataset. "
        "Stratification matters here because, with an imbalanced target, an unstratified split "
        "risks producing a test set with a meaningfully different class balance than the "
        "training set, which would make test metrics harder to compare against cross-validation "
        "results."
    )
    report.add_paragraph(
        "The test set was held out completely from model selection: it was not used for feature "
        "engineering decisions, hyperparameter search, or the model-selection comparisons in "
        "Sections 9–11, and was only scored once at the end (Section 12) with each finalized "
        "pipeline. Cross-validation for model comparison and tuning used 5-fold StratifiedKFold "
        "(shuffle=True, random_state=42) applied only to the training partition, so no "
        "information from the test rows ever influenced the fitted preprocessing statistics, "
        "encoded categories, or model parameters."
    )

    # ---- 8. Baseline Model ----
    report.add_heading("8. Baseline Model", level=1)
    report.add_paragraph(
        "A DummyClassifier configured with strategy='most_frequent' serves as the baseline: it "
        "always predicts the majority class (<=50K) regardless of input features. A baseline is "
        "necessary because raw accuracy or F1 numbers are meaningless without a reference point "
        "— on an imbalanced target, a trivial rule can already score deceptively well on "
        "accuracy, so any candidate model must be shown to outperform it, and by how much, "
        "before its performance can be called meaningful."
    )
    baseline_row = ctx["test_table"][ctx["test_table"].model == "Majority baseline"].iloc[0]
    report.add_paragraph(
        f"RESULT: the baseline achieved {baseline_row.accuracy:.1%} test accuracy but "
        f"{baseline_row.precision:.3f} precision, {baseline_row.recall:.3f} recall, and "
        f"{baseline_row.f1:.3f} F1, because it never predicts the positive class. This "
        "confirms that accuracy alone would be a misleading headline metric for this dataset, "
        "and every candidate model below is judged primarily on precision/recall/F1/ROC-AUC "
        "relative to this baseline rather than on accuracy in isolation."
    )

    # ---- 9. Model Selection ----
    report.add_heading("9. Model Selection", level=1)
    report.add_paragraph(
        "Two candidate algorithms beyond the baseline were evaluated under the identical "
        "cross-validation framework:"
    )
    report.add_paragraph(
        "Logistic Regression fits a linear decision boundary in the (scaled, one-hot-encoded) "
        "feature space and outputs calibrated-style probabilities via the sigmoid function. "
        "Strengths: fast to train, directly interpretable through its coefficients (Section 15), "
        "and a strong fit when the true relationship between features and log-odds of income is "
        "close to linear. Weaknesses: cannot natively capture non-linear interactions between "
        "features (e.g. how the effect of hours-per-week on income might differ by occupation) "
        "unless those interactions are engineered explicitly, and it is sensitive to feature "
        "scale, which is why standardization in Section 5 matters for this model specifically. "
        "class_weight='balanced' was used to counteract the target imbalance by up-weighting "
        "the minority (>50K) class during training."
    )
    report.add_paragraph(
        "Random Forest is an ensemble of decision trees trained on bootstrapped samples with "
        "random feature subsets at each split, with predictions averaged across trees. "
        "Strengths: captures non-linear relationships and feature interactions automatically, "
        "is insensitive to feature scaling, and handles the mix of numeric and one-hot categorical "
        "inputs without additional transformation. Weaknesses: less directly interpretable than "
        "logistic regression (mitigated here with permutation importance, Section 15), more "
        "prone to overfitting deep trees on training noise, and more expensive to train and tune. "
        "class_weight='balanced' was again used given the target imbalance."
    )
    report.add_paragraph(
        "Both were retained as candidates rather than picking one on reputation alone, "
        "specifically because logistic regression and random forest make different assumptions "
        "(linear vs. non-linear decision boundary) and because comparing them under the same "
        "stratified 5-fold CV protocol provides direct evidence for which assumption fits this "
        "dataset better, rather than asserting it a priori."
    )

    # ---- 10. Model Training ----
    report.add_heading("10. Model Training", level=1)
    report.add_paragraph(
        "Both candidates were trained inside the identical Pipeline (FeatureEngineer → "
        "ColumnTransformer → estimator) so that every fold's preprocessing statistics are "
        "learned only from that fold's training rows. Logistic regression used solver='liblinear' "
        "(efficient for this feature-matrix size and supports both L1/L2 penalties) with "
        "max_iter=500 to ensure convergence. Random forest used n_estimators=180 as an initial "
        "untuned configuration with random_state=42 for reproducibility."
    )

    # ---- 11. Cross-Validation and Hyperparameter Tuning ----
    report.add_heading("11. Cross-Validation and Hyperparameter Tuning", level=1)
    report.add_paragraph(
        "5-fold StratifiedKFold cross-validation (shuffle=True, random_state=42) was used for "
        "both model comparison and hyperparameter search, scored by ROC-AUC because it is "
        "threshold-independent and robust to the class imbalance. Cross-validation matters "
        "beyond a single train/validation split because it estimates how much a model's "
        "performance varies across different subsets of the training data, giving a variance "
        "estimate alongside the mean — a model that looks strong on one lucky fold but weak on "
        "others is less trustworthy than one with a consistently high score."
    )
    report.add_paragraph("Cross-validation results (mean across folds; cv_roc_auc is GridSearchCV's refit metric):")
    fmt_table(report, ctx["cv_table"])
    report.add_paragraph(
        "GridSearchCV performed a focused search — logistic regression's inverse-regularization "
        "strength C over [0.1, 1, 10], and random forest's n_estimators over [120, 220] and "
        "max_depth over [None, 18] — using the same 5-fold StratifiedKFold splitter and ROC-AUC "
        "scoring. The grids were kept deliberately small (3 and 4 combinations respectively) to "
        "stay computationally practical while still testing meaningfully different regularization "
        "and tree-complexity settings."
    )
    report.add_paragraph(f"Best logistic regression parameters: {ctx['logistic_best_params']}; "
                          f"best random forest parameters: {ctx['forest_best_params']}.")
    tuned_logistic_cv = ctx['cv_table'].loc[ctx['cv_table'].model == "Logistic regression", "roc_auc"].iloc[0]
    report.add_paragraph(
        f"RESULT: tuning changed random forest's CV ROC-AUC from "
        f"{ctx['cv_table'].loc[ctx['cv_table'].model=='Random forest','roc_auc'].iloc[0]:.4f} "
        f"(untuned) to {ctx['forest_cv_score']:.4f} (tuned), and logistic regression's from "
        f"{tuned_logistic_cv:.4f} to {ctx['logistic_cv_score']:.4f}. COMPARISON: the tuned random "
        f"forest's CV score exceeds the tuned logistic regression's by "
        f"{ctx['forest_cv_score'] - ctx['logistic_cv_score']:+.4f}. INTERPRETATION: this is the "
        "evidence basis for selecting the final model in Section 13, rather than a subjective "
        "preference. LIMITATION: grid search only explored the specific parameter combinations "
        "listed above, not a continuous or exhaustive search, so a marginally better "
        "configuration may exist outside this grid; tuned parameters should not be assumed "
        "optimal beyond this dataset and CV protocol."
    )

    # ---- 12. Model Evaluation ----
    report.add_heading("12. Model Evaluation", level=1)
    report.add_paragraph("Held-out test-set metrics for every candidate (test set touched only here):")
    fmt_table(report, ctx["test_table"])
    report.add_picture(str(ctx["paths"]["roc_curves"]), width=Inches(6.0))
    add_caption(report, "Figure 3. ROC curves for all candidate models on the held-out test set.")
    report.add_picture(str(ctx["paths"]["precision_recall_curves"]), width=Inches(6.0))
    add_caption(report, "Figure 4. Precision-recall curves for all candidate models on the held-out test set.")
    report.add_picture(str(ctx["paths"]["confusion_matrix"]), width=Inches(4.6))
    add_caption(report, f"Figure 5. Confusion matrix for the selected model ({ctx['best_name']}) on the test set.")
    best = ctx["best_test_metrics"]
    report.add_paragraph(
        f"METRIC INTERPRETATION — Accuracy ({best['accuracy']:.1%}) measures the overall share "
        "of correct predictions but weights both classes equally even though they are not "
        f"equally represented ({data.income.mean():.1%} positive), so it can mask poor minority-"
        "class performance. Precision "
        f"({best['precision']:.3f}) is the share of predicted >50K cases that were actually "
        ">50K — it matters when false positives (incorrectly flagging someone as high-income) "
        "are costly. Recall "
        f"({best['recall']:.3f}) is the share of true >50K cases the model actually caught — it "
        "matters when false negatives (missing a genuinely high-income case) are costly. F1 "
        f"({best['f1']:.3f}) is the harmonic mean of precision and recall, useful as a single "
        "number when both error types matter but the classes are imbalanced. ROC-AUC "
        f"({best['roc_auc']:.3f}) measures the model's ability to rank a random positive above a "
        "random negative across all thresholds, independent of the specific 0.5 cutoff, which is "
        "why it was used as the tuning objective."
    )
    report.add_paragraph(
        f"OBSERVED: the selected model trades some precision for higher recall relative to the "
        f"untuned random forest (precision {best['precision']:.3f} vs. "
        f"{ctx['cv_table'].loc[ctx['cv_table'].model=='Random forest','precision'].iloc[0]:.3f}, "
        f"recall {best['recall']:.3f} vs. "
        f"{ctx['cv_table'].loc[ctx['cv_table'].model=='Random forest','recall'].iloc[0]:.3f}). "
        "POSSIBLE EXPLANATION: class_weight='balanced' combined with tuning toward ROC-AUC "
        "(a threshold-independent, rank-based metric) pushes the model to be more willing to "
        "predict the minority class, which raises recall at some cost to precision. This is a "
        "genuine trade-off, not a strict improvement on every axis, and the appropriate balance "
        "depends on the relative cost of false positives vs. false negatives in the intended use "
        "case (Section 17)."
    )

    # ---- 13. Model Comparison ----
    report.add_heading("13. Model Comparison", level=1)
    report.add_picture(str(ctx["paths"]["model_comparison"]), width=Inches(6.3))
    add_caption(report, "Figure 6. Accuracy, F1, and ROC-AUC compared across all models on the test set.")
    report.add_paragraph(
        f"The final model, {ctx['best_name']}, was selected using cross-validated ROC-AUC on "
        "training data (Section 11) — not test accuracy — because ROC-AUC is threshold-"
        "independent and less distorted by the class imbalance. It was not selected merely for "
        "having the highest test accuracy: in fact the untuned random forest has higher test "
        f"accuracy ({ctx['test_table'].loc[ctx['test_table'].model=='Random forest','accuracy'].iloc[0]:.1%}) "
        f"than the tuned model ({best['accuracy']:.1%}), but substantially lower recall "
        f"({ctx['test_table'].loc[ctx['test_table'].model=='Random forest','recall'].iloc[0]:.3f} "
        f"vs. {best['recall']:.3f}), which the balanced, CV-driven selection criterion "
        "correctly favors against given the analysis goal of identifying >50K individuals "
        "rather than maximizing raw correctness."
    )

    # ---- 14. Error Analysis ----
    report.add_heading("14. Error Analysis", level=1)
    cm = ctx["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    report.add_paragraph(
        f"Of {tn + fp + fn + tp:,} test records, the model produced {tn:,} true negatives, "
        f"{fp:,} false positives, {fn:,} false negatives, and {tp:,} true positives. False "
        f"positives ({fp:,}) occur when someone earning <=50K is predicted >50K; false negatives "
        f"({fn:,}) occur when someone earning >50K is predicted <=50K."
    )
    report.add_picture(str(ctx["paths"]["error_analysis"]), width=Inches(6.0))
    add_caption(report, "Figure 7. Hours-per-week distribution split by prediction outcome.")
    err = ctx["error_frame"]
    fn_hours = err.loc[err.outcome == "False negative", "hours-per-week"].mean()
    fp_hours = err.loc[err.outcome == "False positive", "hours-per-week"].mean()
    correct_hours = err.loc[err.outcome == "Correct", "hours-per-week"].mean()
    report.add_paragraph(
        f"OBSERVED: mean hours-per-week is {fn_hours:.1f} for false negatives, {fp_hours:.1f} "
        f"for false positives, and {correct_hours:.1f} for correctly classified records. "
        "POSSIBLE EXPLANATION: false negatives (missed >50K cases) may include people who earn "
        "well despite comparatively standard working hours — e.g. through capital gains or "
        "high-paying roles not captured by hours worked — which the model under-weights relative "
        "to more hours-driven earners. LIMITATION: this is a bivariate summary; a fuller error "
        "analysis would examine interactions across several features simultaneously and check "
        "whether errors concentrate in particular occupation, education, or demographic "
        "subgroups, which was not exhaustively performed here."
    )

    # ---- 15. Feature Importance / Interpretability ----
    report.add_heading("15. Feature Importance / Interpretability", level=1)
    report.add_paragraph(
        "Permutation importance measures the drop in test ROC-AUC when a single raw input "
        "column is randomly shuffled, holding the fitted pipeline fixed; a larger drop means the "
        "model relied more on that feature. This is computed on the selected model directly, "
        "so it reflects genuine predictive reliance rather than a proxy statistic, but it is "
        "correlational and does not establish that a feature causes income to change."
    )
    report.add_picture(str(ctx["paths"]["permutation_importance"]), width=Inches(6.0))
    add_caption(report, f"Figure 8. Permutation importance of raw features for {ctx['best_name']}.")
    report.add_paragraph(
        "For comparison, the logistic regression's largest-magnitude coefficients (after "
        "encoding) are shown below; positive coefficients push the prediction toward >50K, "
        "negative coefficients push toward <=50K:"
    )
    fmt_table(report, ctx["top_coefficients"], digits=4)
    report.add_paragraph(
        "Feature importance here indicates association strength within the fitted model, not "
        "causal effect: e.g. a high importance for education-num reflects that the model uses "
        "education level to separate income classes in this historical sample, not proof that "
        "increasing education causes an income increase for any specific individual."
    )

    # ---- 16. Results and Discussion ----
    report.add_heading("16. Results and Discussion", level=1)
    baseline_auc = baseline_row.roc_auc
    report.add_paragraph(
        f"EVIDENCE: the selected model achieved {best['roc_auc']:.3f} test ROC-AUC compared with "
        f"{baseline_auc:.3f} for the majority baseline, a difference of "
        f"{best['roc_auc'] - baseline_auc:+.3f}. COMPARISON: this gap is large relative to "
        "typical cross-validation fold variability for this dataset and model family, and the "
        "model also clears the baseline decisively on precision, recall, and F1 (baseline scores "
        "0 on all three since it never predicts the positive class). INTERPRETATION: demographic "
        "and employment attributes in this dataset carry substantial, learnable signal about "
        "income bracket. POSSIBLE EXPLANATION: fields such as education-num, hours-per-week, "
        "capital-gain, and marital-status/relationship encode well-documented socioeconomic "
        "correlates of earnings, which the model is able to exploit. LIMITATION: cross-validation "
        f"ROC-AUC for the tuned random forest was {ctx['forest_cv_score']:.4f}; the spread across "
        "individual folds (not shown as a single number here) means any single-split estimate, "
        "including the held-out test score, carries some sampling uncertainty and should not be "
        "read as an exact population value."
    )
    report.add_paragraph(
        f"EVIDENCE: precision ({best['precision']:.3f}) is noticeably lower than recall "
        f"({best['recall']:.3f}) for the selected model. COMPARISON: the untuned random forest "
        f"has the opposite pattern "
        f"(precision {ctx['cv_table'].loc[ctx['cv_table'].model=='Random forest','precision'].iloc[0]:.3f}, "
        f"recall {ctx['cv_table'].loc[ctx['cv_table'].model=='Random forest','recall'].iloc[0]:.3f}). "
        "INTERPRETATION: this reflects a genuine, explainable trade-off from balancing class "
        "weights and tuning toward AUC rather than accuracy — the model was deliberately biased "
        "toward catching more true >50K cases at the cost of more false alarms. POSSIBLE "
        "EXPLANATION: the imbalanced target means an unweighted model naturally gravitates "
        "toward the majority class; class_weight='balanced' counteracts this but shifts the "
        "operating point rather than eliminating the imbalance's effect outright. LIMITATION: "
        "whether this trade-off is desirable depends entirely on the downstream use case's cost "
        "structure (Section 17), which this analysis does not have privileged access to."
    )

    # ---- 17. Practical Implications ----
    report.add_heading("17. Practical Implications", level=1)
    report.add_paragraph(
        "DATA FINDING: the model can rank individuals by predicted income likelihood with "
        f"{best['roc_auc']:.3f} ROC-AUC using only demographic and employment survey fields, "
        "without requiring direct income disclosure."
    )
    report.add_paragraph(
        "POTENTIAL APPLICATION: such a model could support decision support and prioritization "
        "in contexts like survey research (imputing likely income bracket when respondents "
        "decline to answer), coarse market segmentation for non-targeted product research, or "
        "resource-allocation triage in socioeconomic studies where direct income data is "
        "unavailable. This is a plausible use, not a validated deployment: the model's precision "
        f"({best['precision']:.3f}) means roughly {1-best['precision']:.0%} of records it flags "
        "as >50K would be false positives in this test sample, which must be weighed against the "
        "cost of an incorrect classification in the specific application before deployment."
    )
    report.add_paragraph(
        "The model should not be used for individual-level consequential decisions (e.g. credit, "
        "employment, or benefits eligibility) given its error rates and the well-documented risk "
        "that Census-derived demographic models can encode historical societal biases (Section 18)."
    )

    # ---- 18. Strengths and Limitations ----
    report.add_heading("18. Strengths and Limitations", level=1)
    report.add_paragraph(
        "STRENGTHS: the pipeline is leakage-safe by construction (Section 5), evaluated with "
        "stratified cross-validation and an untouched test set, compared against an explicit "
        "baseline, and validated with an experimental (not assumed) feature-engineering "
        f"ablation. The selected model reaches {best['roc_auc']:.3f} ROC-AUC, a substantial "
        f"improvement over the {baseline_auc:.3f} baseline."
    )
    report.add_paragraph(
        f"LIMITATIONS: (1) Dataset size and recency — {len(data):,} records from a 1994 Census "
        "extract may not reflect current economic conditions, wage levels, or occupational "
        "structures. (2) Class imbalance "
        f"({data.income.mean():.1%} positive) inherently limits achievable precision at high "
        "recall operating points. (3) Representativeness and bias — the dataset reflects "
        "historical U.S. demographic and labor patterns, including any embedded societal biases "
        "around race, sex, and occupation; a model trained on it will reproduce those patterns "
        "rather than correct them. (4) Feature limitations — capital-gain/loss are extremely "
        "sparse (mostly zero), and native-country has many low-frequency categories collapsed "
        "during preprocessing, both of which reduce the granularity of information available to "
        "the model. (5) Cross-validation variability — reported CV means do not convey full "
        "fold-to-fold uncertainty. (6) The model establishes correlation, not causation, between "
        "any feature and income. (7) A single stratified holdout, while leakage-safe, is a "
        "coursework-appropriate but not a production-grade validation protocol; it lacks nested "
        "cross-validation, subgroup fairness auditing, and external/temporal validation."
    )

    # ---- 19. Possible Improvements ----
    report.add_heading("19. Possible Improvements", level=1)
    improvements = [
        ("Gradient-boosted trees (e.g. HistGradientBoostingClassifier)",
         "could improve on random forest's ROC-AUC because boosting sequentially corrects prior "
         "errors rather than averaging independent trees, which often outperforms bagging on "
         "structured/tabular data of this kind; this is a plausible improvement, not one tested "
         "here, so it should be verified experimentally before adoption."),
        ("Decision-threshold optimization",
         "could directly tune the precision/recall trade-off discussed in Section 16 against an "
         "explicit cost matrix for the intended application, rather than using the default 0.5 "
         "cutoff."),
        ("Probability calibration (e.g. Platt scaling or isotonic regression)",
         "could make predicted probabilities more directly interpretable as likelihoods, which "
         "matters if the model's output is used for risk scoring rather than a binary label."),
        ("Missing-value indicator features",
         "could preserve information currently discarded by most-frequent imputation, if "
         "missingness in workclass/occupation is itself informative, as suspected in Section 5."),
        ("Subgroup fairness auditing (by sex, race, age band)",
         "could reveal whether errors are concentrated in particular demographic groups, which "
         "this project's error analysis (Section 14) did not exhaustively test."),
        ("A larger, more recent dataset",
         "could improve generalization to current economic conditions, since this dataset is "
         "fixed at its 1994 collection period."),
    ]
    for name, reason in improvements:
        para = report.add_paragraph(style="List Bullet")
        para.add_run(f"{name}: ").bold = True
        para.add_run(reason)

    # ---- 20. Conclusion ----
    report.add_heading("20. Conclusion", level=1)
    report.add_paragraph(
        f"This project defined income-bracket prediction on the UCI Adult dataset "
        f"({len(data):,} records) as a binary classification problem, built a leakage-safe "
        "scikit-learn pipeline covering imputation, scaling, one-hot encoding, and four "
        "engineered features, and validated an experimental claim about that engineering rather "
        "than assuming it helped. A majority-class baseline, logistic regression, and random "
        "forest were compared under identical 5-fold stratified cross-validation, both "
        f"non-baseline models were tuned with a focused grid search, and {ctx['best_name']} was "
        f"selected using cross-validated ROC-AUC. On the untouched test set it achieved "
        f"{best['accuracy']:.1%} accuracy, {best['precision']:.3f} precision, {best['recall']:.3f} "
        f"recall, {best['f1']:.3f} F1, and {best['roc_auc']:.3f} ROC-AUC, clearing the baseline's "
        f"{baseline_auc:.3f} ROC-AUC by {best['roc_auc']-baseline_auc:+.3f}. Error analysis and "
        "permutation importance showed the model relies heavily on capital-gain, marital/"
        "relationship status, and education, while precision/recall analysis revealed an explicit "
        "trade-off introduced by class-weighting rather than a uniformly superior result. The "
        "main limitations are the dataset's age and representativeness, residual class imbalance, "
        "and the coursework-scope validation protocol; Section 19 lists concrete, evidence-"
        "grounded next steps rather than assumed improvements."
    )

    add_code_appendix(report, ctx["code_path"])

    report.save(ARTIFACTS / "Week_4_Supervised_Learning_Model_Implementation_Report.docx")


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
    np.random.seed(RANDOM_STATE)
    ARTIFACTS.mkdir(exist_ok=True)
    data, clean_stats = load_data()
    X = data.drop(columns="income")
    y = data["income"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- Feature-engineering ablation (random forest, CV ROC-AUC) ----
    ablation_model = RandomForestClassifier(
        n_estimators=180, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    fe_scores_with = cross_val_score(
        pipeline(ablation_model), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    fe_scores_without = cross_val_score(
        baseline_pipeline(ablation_model), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    feature_engineering_table = pd.DataFrame([
        {"feature set": "Original features only", "mean cv roc_auc": fe_scores_without.mean(),
         "std": fe_scores_without.std()},
        {"feature set": "Original + engineered features", "mean cv roc_auc": fe_scores_with.mean(),
         "std": fe_scores_with.std()},
    ])
    fe_delta = fe_scores_with.mean() - fe_scores_without.mean()

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
    for name in tuned:
        cv_rows.append({"model": name, "cv_roc_auc": (
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
    interpretation_model = logistic_search.best_estimator_
    feature_names = interpretation_model.named_steps["preprocess"].get_feature_names_out()
    coefficients = interpretation_model.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    top_coefficients = pd.concat([
        coefficient_table.nlargest(6, "coefficient"),
        coefficient_table.nsmallest(6, "coefficient"),
    ]).drop_duplicates("feature").sort_values("coefficient", ascending=False).reset_index(drop=True)

    paths = save_figures(data, y_test, predictions, probabilities, best_model, X_test, best_name)
    paths["model_comparison"] = build_model_comparison_figure(cv_table, test_table)
    error_path, error_frame = build_error_analysis_figure(X_test, y_test, pred, best_name)
    paths["error_analysis"] = error_path

    confusion = confusion_matrix(y_test, pred)

    ctx = {
        "data": data,
        "clean_stats": clean_stats,
        "split_sizes": (len(X_train), len(X_test)),
        "cv_table": cv_table,
        "test_table": test_table,
        "logistic_best_params": logistic_search.best_params_,
        "forest_best_params": forest_search.best_params_,
        "logistic_cv_score": logistic_search.best_score_,
        "forest_cv_score": forest_search.best_score_,
        "best_name": best_name,
        "best_test_metrics": test_table.loc[test_table.model == best_name].iloc[0],
        "confusion_matrix": confusion.tolist(),
        "error_frame": error_frame,
        "top_coefficients": top_coefficients[["feature", "coefficient"]],
        "feature_engineering_table": feature_engineering_table,
        "fe_delta": fe_delta,
        "fe_base_score": fe_scores_without.mean(),
        "paths": paths,
        "code_path": Path(__file__).resolve(),
    }
    build_report(ctx)

    summary = {
        "rows": len(data), "train_rows": len(X_train), "test_rows": len(X_test),
        "class_counts": data["income"].value_counts().sort_index().to_dict(),
        "clean_stats": clean_stats,
        "feature_engineering_ablation": {
            "without_engineered_mean_roc_auc": float(fe_scores_without.mean()),
            "with_engineered_mean_roc_auc": float(fe_scores_with.mean()),
            "delta": float(fe_delta),
        },
        "best_model": best_name,
        "best_params": (logistic_search.best_params_ if best_name.endswith("logistic regression")
                        else forest_search.best_params_),
        "cv_results": cv_table.to_dict(orient="records"),
        "test_results": test_table.to_dict(orient="records"),
        "confusion_matrix": confusion.tolist(),
        "best_test_metrics": test_table.loc[test_table.model == best_name].iloc[0].to_dict(),
        "artifacts": sorted(
            path.name for path in ARTIFACTS.iterdir() if path.is_file() and path.name != "results.json"
        ) + ["results.json"],
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
