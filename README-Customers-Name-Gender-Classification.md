# Name → Gender Classification

> Predicting customer gender from free-text first names — applied machine learning on noisy, real-world text with a trained, saved model.

**Context:** Customer records often capture a name but not gender, yet gender breakdowns matter for segmentation, reporting, and program design. This project builds a model that infers likely gender directly from a name.

---

## What's in this repo

| File | What it is |
|---|---|
| `Gender Identification- Name based.ipynb` | The full analysis and modelling notebook (read this first) |
| `name_classification_xgb.pkl` | Trained XGBoost classifier (first iteration) |
| `name_classification_xgb2.pkl` | Retrained / improved XGBoost classifier (second iteration) |

## Approach

- **Problem framing:** treat gender prediction as a supervised text-classification task over names.
- **Feature engineering:** derive predictive signal from the structure of names (character patterns, suffixes, and substrings) rather than relying on a fixed lookup table — so the model can handle names it has never seen.
- **Model:** an [XGBoost](https://xgboost.readthedocs.io/) gradient-boosted classifier. Two saved model versions (`xgb` and `xgb2`) capture an iteration cycle — train, evaluate, refine, retrain.
- **Reusability:** the trained models are serialized to `.pkl` so they can be loaded and applied to new name lists without retraining.

## Why this is a useful work sample

- **Applied ML on imperfect input** — real name fields are noisy, abbreviated, and culturally varied; this handles that rather than a clean textbook dataset.
- **Feature thinking** — engineering signal from raw text is the part that separates a copied tutorial from real modelling work.
- **Iteration and evaluation** — two model versions show a deliberate improvement loop, not a single lucky run.
- **Deployability** — saving the model artifact means the work can actually be used downstream, not just demonstrated once.

---

*Author: Denis Mwaniki.*
