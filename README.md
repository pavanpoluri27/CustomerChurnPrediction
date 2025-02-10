# Customer Churn Prediction using Machine Learning

## Overview

This project implements a ML model to predict customer churn for a telecommunications company. Customer churn, is a critical business metric, and accurately predicting which customers are likely to churn allows the company to proactively intervene and retain valuable customers. 
The project leverages a Telco Customer Churn (Kaggle) dataset and builds a Logistic Regression model with feature engineering and hyperparameter tuning (GridSearchCV).
This project represents my initial foray into the field of ML. Through this experience, I gained valuable skills in data preprocessing, feature engineering, model selection, and evaluation. I'm excited to continue learning and growing in this field.

## Dataset

*   **Source:** The dataset is the "Telco Customer Churn" dataset, found on Kaggle.
*   **Description:** The dataset contains information about telecommunications customers, including demographic information, service usage details, contract information, and churn status.

## Key Steps

1.  **Data Loading and Preprocessing:**
    *   Loaded the Telco Customer Churn dataset using pandas.
2.  **Feature Engineering:**
    *   Created new features to capture more complex relationships.
3.  **Data Preprocessing Pipeline:**
    *   Used a `ColumnTransformer` to apply different preprocessing steps to numeric and categorical features.
    *   Numeric features were imputed using the median and scaled using `StandardScaler`.
    *   Categorical features were imputed with "missing" and one-hot encoded using `OneHotEncoder` with `handle_unknown='ignore'`.
4.  **Hyperparameter Tuning:**
    *   Performed hyperparameter tuning for the Logistic Regression model using `GridSearchCV`.
    *   Tuned parameters such as C (regularization strength), penalty type (l1, l2, elasticnet), and solver.
    *   Used a scoring of `accuracy` to select the optimal hyperparameters.
5.  **Model Training:**
    *   Trained a `LogisticRegression` model with `class_weight='balanced'` to address class imbalance in the target variable, using the *best* hyperparameters from the tuning process.
    *   The input data set was split into train & test segments and the model was trained on the train segment of the input data set.
6.  **Model Evaluation:**
    *   Evaluated the model's performance on the test segement of the input data set using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
    *   Visualized the confusion matrix and ROC curve to assess the model's performance.
7.  **Feature Importance Analysis:**
    *   Extracted the coefficients from the trained `LogisticRegression` model.
    *   Created a DataFrame to associate the coefficients with the corresponding feature names.
    *   Visualized the most important features based on the absolute value of their coefficients.
    *   Great care was taken to avoid data leakage during feature importance analysis.

## Key Libraries and Tools

*   Python
*   pandas
*   scikit-learn
*   matplotlib
*   seaborn

## Results

The Logistic Regression model, with optimized hyperparameters, achieved an accuracy of **72.99%** on the test set. Feature importance analysis revealed the most influential factors driving customer churn.

*NOTE: The project has gone through a careful process of iterative feature engineering and hyperparameter tuning. It has been shown that new combinations of columns and tuning parameters can be powerful predictors of customer churn.*

## Data Leakage Prevention

Data cleaning, transforming, feature engineering and column extraction were all performed only on the training set to prevent data leakage.

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/pavanpoluri27/CustomerChurnPrediction.git
    ```

2.  **Install dependencies:**

    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

3.  **Run the notebook:**

    ```bash
    jupyter notebook CustomerChurnPrediction.ipynb
    ```

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.

## Contact

[Pavan Poluri] - [pavanpoluri@gmail.com] - [www.linkedin.com/in/pavan-poluri]
