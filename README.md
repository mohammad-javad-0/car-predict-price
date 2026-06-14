# Car Price Prediction Pipeline

An end-to-end Machine Learning project designed to predict used car prices using advanced data preprocessing and optimized regression architectures. This repository covers everything from rigorous Exploratory Data Analysis (EDA) and feature engineering to automated robust scaling, hyperparameter tuning, and final production model deployment.

---

## Project Architecture & Workflow

The project is split into two primary components to ensure clean execution and prevent data leakage:

1. **Exploratory Data Analysis (`eda.ipynb`)**: Understanding distributions, correlation analysis, feature extraction, and identifying target anomalies.

2. **Model Training Pipeline (`training.ipynb`)**: Constructing data pipelines, hyperparameter tuning via cross-validation, and production model serialization.

---
## Dataset Setup

The dataset used in this project was obtained from Kaggle:

**Dataset Source:**  
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

This dataset contains used car information collected from online platforms and includes vehicle specifications, ownership details, engine features, and pricing information. The `car details v4.csv` file from this dataset was used for this project.

### How to Use the Dataset

1. Download the dataset from the Kaggle link above.
2. Extract the file:

```
car details v4.csv
```

3. Place the file inside the following directory:

```
data/car details v4.csv
```

The final project structure should look like this:

```
car-price-prediction/
│
├── data/
│   └── car details v4.csv
│
├── source_code/
│   ├── eda.ipynb
│   └── training.ipynb
│
├── model/
│   └── final_model.pkl
│
├── requirements.txt
└── README.md
```

The notebooks load the dataset from the following path:

```python
pd.read_csv('data/car details v4.csv')
```

Please refer to the original Kaggle dataset page for the dataset license and usage terms. The dataset is published under the Open Database License.

---

# 1. Exploratory Data Analysis (EDA) & Feature Engineering

During the EDA phase, the raw dataset (`car details v4.csv`) underwent extensive auditing to transform noisy string data into high-value numerical features:

- **Engine Capacity**:
  - Removed unit markers (`cc`)
  - Converted values into float format

- **Regex Feature Extraction**:
  - Extracted `Max_Power_bhp` and `Max_Power_rpm` from the complex `Max Power` column.
  - Extracted `Max_Torque_Nm` and `Max_Torque_rpm` from the noisy `Max Torque` column.

- **Missing Value Analysis**:
  - Used `missingno` matrix and heatmap visualizations to analyze missing data patterns.

- **Anomalies & Outliers**:
  - Detected right-skewed distributions in features such as `Price`, `Kilometer`, and `Engine`.
  - Extreme values were analyzed using the IQR method.
  - Outliers were retained because they represented real high-performance luxury vehicles rather than data entry errors.

---

# 2. Data Preprocessing & Pipeline Construction

To guarantee data integrity and prevent **Data Leakage**, all transformations were wrapped inside `scikit-learn` Pipelines and executed dynamically within a 5-Fold Cross-Validation process.

Two different preprocessing pipelines were created based on model requirements:

---

## A. Pipeline for Linear & Distance-Based Models

Models:
- SVR
- Linear Regression
- ElasticNet

### Categorical Features

```
SimpleImputer(strategy='most_frequent')
        ↓
OneHotEncoder(handle_unknown='ignore')
```

### Numerical Features

```
SimpleImputer(strategy='median')
        ↓
RobustScaler()
        ↓
PowerTransformer(method='yeo-johnson')
```

---

## B. Pipeline for Tree-Based Models

Models:
- Random Forest
- Gradient Boosting
- AdaBoost

### Categorical Features

- Missing value imputation
- One-hot encoding

### Numerical Features

```
SimpleImputer(strategy='median')
```

Tree-based models are naturally resistant to scaling and feature skewness.

---

> **Target Transformation**
>
> The target variable `Price` was highly skewed.
> A `TransformedTargetRegressor` wrapper was used with `log1p` transformation during training and automatic inverse transformation using `expm1` during evaluation.

---

# 3. Model Training & Evaluation Metrics

The training process consisted of two phases:

## Phase 1: Base Model Benchmarking (5-Fold CV)

Six regression models were evaluated:

| Model | Status |
|---|---|
| **Random Forest Regressor** | Candidate Selected |
| **Gradient Boosting Regressor** | Candidate Selected |
| **Support Vector Regressor (SVR)** | Candidate Selected |
| AdaBoost Regressor | Excluded |
| ElasticNet | Excluded |
| Linear Regression | Failed / Abandoned |

---

## Phase 2: Hyperparameter Tuning via `RandomizedSearchCV`

The top 3 models were optimized using a 30-iteration randomized search.

After comparing tuned RMSE scores, **Support Vector Regressor (SVR)** achieved the best performance.

---

# Final Champion Model Performance

The optimized SVR model was evaluated on a completely isolated testing split (`X_test`):

- **Mean Absolute Error (MAE)**:
```
245188.19061755016
```

- **Root Mean Squared Error (RMSE)**:
```
703483.8934994731
```

- **R² Score**:
```
0.9162243271143017
```

- **Mean Absolute Percentage Error (MAPE)**:
```
0.13257586126160392
```

---

## Selected Best Hyperparameters

```python
{
    'kernel': 'rbf',
    'C': 3.593813663804626,
    'gamma': 0.01,
    'degree': 4
}
```

The production pipeline was refit on the complete dataset (`X`, `y`) using these optimized hyperparameters and saved as:

```
model/final_model.pkl
```

---

# How to Run & Reproduce

## 1. Clone Repository

```bash
git clone https://github.com/mohammad-javad-0/car-predict-price.git

cd car-price-prediction
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Execute Pipelines

- Open:

```
source_code/eda.ipynb
```

to review EDA and feature engineering.

- Run:

```
source_code/training.ipynb
```

to execute model training, cross-validation, tuning, and export the final model.
