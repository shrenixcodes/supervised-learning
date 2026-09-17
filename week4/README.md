# Week 4 — Supervised learning on UCI Adult

This project predicts whether an individual's annual income exceeds `$50K` using the
[UCI Adult/Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult).
`adult_supervised_learning.py` downloads the data programmatically, applies leakage-safe
feature engineering and preprocessing inside scikit-learn pipelines, compares a majority
baseline with logistic regression and random forest models, tunes the candidates, and
writes a DOCX report with executed metrics, error analysis, interpretability, and figures.

## Reproduce

```bash
python -m pip install -r week4/requirements.txt
python week4/adult_supervised_learning.py
```

The script writes `week4/artifacts/Week_4_Supervised_Learning_Model_Implementation_Report.docx` and PNG figures. Downloads are
kept only in a local, git-ignored cache (`week4/.download_cache/`) so reruns remain fast
without committing data.

The split is stratified and fixed (`random_state=42`). All imputation, scaling, one-hot
encoding, and feature engineering are fitted only on training folds through a pipeline.
