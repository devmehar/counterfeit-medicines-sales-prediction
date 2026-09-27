# Counterfeit Medicines Sales Prediction

A regression model that predicts counterfeit medicine sales volume from structured pharma/retail data, supporting anti-counterfeiting monitoring and resource allocation.

## Problem Statement

Counterfeit medicines are a serious public health and revenue risk. Being able to predict where and how much counterfeit sales volume to expect — based on medicine type, distribution area, and other attributes — helps regulators and pharma companies prioritize inspection and enforcement efforts before losses grow.

## Demo

*(Add a screenshot here — e.g., a small table of sample inputs next to actual vs. predicted `Counterfeit_Sales`, or a feature-importance bar chart from the notebook.)*

## Dataset

- Source: Counterfeit Medicines Sales dataset — *link the exact source you used*
- Features include medicine ID/type, distribution area, counterfeit weight, and other categorical/numerical attributes
- Target: `Counterfeit_Sales` (continuous — a regression problem)

## Approach

1. **Preprocessing:** Imputed missing `Counterfeit_Weight` values with the column mean, dropped the non-predictive `Medicine_ID` column, and one-hot encoded remaining categorical features.
2. **Modeling — Decision Tree:** Tuned a `DecisionTreeRegressor` via `RandomizedSearchCV` (10-fold CV, MAE scoring) over depth, split, and leaf-size hyperparameters.
3. **Modeling — Random Forest:** Repeated the same tuning process with a `RandomForestRegressor` to compare ensemble performance against the single tree.
4. **Evaluation:** Compared both tuned models using mean absolute error (MAE) and a normalized accuracy-like score relative to the target's value range, plus feature importance analysis to identify the strongest predictors.

## Results

| Model | MAE | Notes |
|---|---|---|
| Decision Tree (tuned) | *add your result* | Best hyperparameters found via `RandomizedSearchCV` |
| Random Forest (tuned) | *add your result* | Ensemble of tuned decision trees |

*(Tip: state which model won and by how much, and name the top 2–3 most important features from the feature-importance analysis — recruiters like seeing that you interpret the model, not just report a number.)*

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## How to Run

```bash
git clone https://github.com/<your-username>/counterfeit-medicines-sales-prediction.git
cd counterfeit-medicines-sales-prediction
pip install -r requirements.txt
```

1. Download the dataset (link above) and place the train/test CSVs in a `data/` folder.
2. Open `ML_P3_Counterfeit_Medicines_Sales_Prediction.ipynb` in Jupyter or Colab.
3. Run all cells in order — the `RandomizedSearchCV` steps may take a few minutes on CPU.

## Project Structure

```
├── ML_P3_Counterfeit_Medicines_Sales_Prediction.ipynb
├── requirements.txt
├── README.md
└── data/                  # not included — see Dataset section
```

## Future Improvements

- Add gradient boosting models (XGBoost/LightGBM) for comparison against Decision Tree/Random Forest
- Engineer interaction features (e.g., medicine type × distribution area) to capture regional counterfeit patterns
- Use SHAP values instead of built-in feature importances for more reliable, model-agnostic interpretation
- Package the best model behind a simple API/demo for "predict sales given new inputs" use cases

## License

This project is licensed under the MIT License.
