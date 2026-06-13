```markdown

\# Car Price Prediction Pipeline



An end-to-end Machine Learning project designed to predict used car prices using advanced data preprocessing and optimized regression architectures. This repository covers everything from rigorous Exploratory Data Analysis (EDA) and feature engineering to automated robust scaling, hyperparameter tuning, and final production model deployment.



\---



\## Project Architecture \& Workflow



The project is split into two primary components to ensure clean execution and prevent data leakage:



1\. \*\*Exploratory Data Analysis (`eda.ipynb`)\*\*: Understanding distributions, correlation analysis, feature extraction, and identifying target anomalies.

2\. \*\*Model Training Pipeline (`training.ipynb`)\*\*: Constructing data pipelines, hyperparameter tuning via cross-validation, and production model serialization.



\---



\## 1. Exploratory Data Analysis (EDA) \& Feature Engineering



During the EDA phase, the raw dataset (`car details v4.csv`) underwent extensive auditing to transform noisy string data into high-value numerical signatures:



\* \*\*Engine Capacity\*\*: Stripped unit markers (`cc`) and converted values into float formats.

\* \*\*Regex Feature Extraction\*\*: 

&#x20; \* Extracted `Max\_Power\_bhp` and `Max\_Power\_rpm` from the complex `Max Power` text column.

&#x20; \* Extracted `Max\_Torque\_Nm` and `Max\_Torque\_rpm` from the noisy `Max Torque` text column.

\* \*\*Missing Value Analysis\*\*: Utilized `missingno` (matrix and heatmap plots) to visualize missing data dependencies.

\* \*\*Anomalies \& Outliers\*\*: Detected significant right-skewness in features like `Price`, `Kilometer`, and `Engine`. High-leverage extreme values were audited using the IQR method. These were safely \*\*retained\*\* as they accurately represent high-performance luxury vehicles rather than entry errors.



\---



\## 2. Data Preprocessing \& Pipeline Construction



To guarantee absolute integrity and prevent \*\*Data Leakage\*\*, all transformations were wrapped securely inside `scikit-learn` Pipelines and executed dynamically within a 5-Fold Cross-Validation loop. 



We engineered separate preprocessing pipelines based on model architectural demands:



\### A. Pipeline for Linear \& Distance-Based Models (SVR, Linear Regression, ElasticNet)

\* \*\*Categorical Features\*\*: `SimpleImputer(strategy='most\_frequent')` $\\rightarrow$ `OneHotEncoder(handle\_unknown='ignore')`

\* \*\*Numerical Features\*\*: `SimpleImputer(strategy='median')` $\\rightarrow$ `RobustScaler()` (to mitigate outlier leverage) $\\rightarrow$ `PowerTransformer(method='yeo-johnson')` (to correct feature skewness).



\### B. Pipeline for Tree-Based Models (Random Forest, Gradient Boosting, AdaBoost)

\* \*\*Categorical Features\*\*: Standard Imputation \& One-Hot Encoding.

\* \*\*Numerical Features\*\*: `SimpleImputer(strategy='median')` (Tree models are naturally invariant to scaling and skewness).



> \*\*Target Transformation\*\*: The target variable `Price` was highly skewed. We wrapped our pipelines inside a `TransformedTargetRegressor` using a $log1p$ function to log-transform the target during training and automatically back-transform predictions via $expm1$ during evaluation.



\---



\## 3. Model Training \& Evaluation Metrics



We performed a dual-phase training strategy: Benchmarking default architectures followed by localized Hyperparameter Optimization.



\### Phase 1: Base Model Benchmarking (5-Fold CV)

We cross-validated six base regressors. \*\*Linear Regression\*\* collapsed under complex multi-collinearity and was dropped due to extreme scaling errors. The remaining top performers were evaluated across $R^2$, $MAE$, and $RMSE$:



| Model | Status |

| :--- | :--- |

| \*\*Random Forest Regressor\*\* | Candidate Selected |

| \*\*Gradient Boosting Regressor\*\* | Candidate Selected |

| \*\*Support Vector Regressor (SVR)\*\* | Candidate Selected |

| AdaBoost Regressor | Excluded |

| ElasticNet | Excluded |

| Linear Regression | Failed / Abandoned |



\### Phase 2: Hyperparameter Tuning via `RandomizedSearchCV`

The top 3 models were subjected to a 30-iteration randomized search over a wide hyperparameter space. After evaluating the tuned `RMSE` scores, the \*\*Support Vector Regressor (SVR)\*\* achieved the superior global minimum error.



\---



\## Final Champion Model Performance



The optimized SVR configuration was evaluated on a completely isolated testing split (`X\_test`), yielding highly robust and stable metric evaluations:



\* \*\*Mean Absolute Error (MAE)\*\*: `\[245188.19061755016]`

\* \*\*Root Mean Squared Error (RMSE)\*\*: `\[703483.8934994731]`

\* \*\*$R^2$ Score\*\*: `\[0.9162243271143017]`

\* \*\*Mean Absolute Percentage Error (MAPE)\*\*: `\[0.13257586126160392]`



\### Selected Best Hyperparameters:

```python

{

&#x20;   'kernel': 'rbf',

&#x20;   'C': 3.593813663804626,

&#x20;   'gamma': 0.01,

&#x20;   'degree': 4

}



```



The production pipeline was refit on the \*\*entire dataset ($X, y$)\*\* using these optimized hyperparameters and serialized to `model/final\_model.pkl` for deployment.



\---



\## How to Run \& Reproduce



1\. \*\*Clone the repository\*\*:

```bash

git clone \https://github.com/mohammad-javad-0/car-predict-price.git
cd car-price-prediction



```





2\. \*\*Activate your Virtual Environment\*\*:

```bash

source venv/Scripts/activate  # On Git Bash / Linux



```





3\. \*\*Install Dependencies\*\*:

```bash

pip install -r requirements.txt



```





4\. \*\*Execute Pipelines\*\*:

\* Open `source\_code/eda.ipynb` to review features and output cleaned data.

\* Run `source\_code/training.ipynb` to execute the full cross-validation grid and export the final model.







```

