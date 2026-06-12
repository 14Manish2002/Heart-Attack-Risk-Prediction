# ❤️ Heart Disease Risk Prediction

## Overview

This project presents an end-to-end machine learning framework for predicting heart disease risk using demographic, lifestyle, and clinical health indicators. The objective is to identify individuals at high risk of developing heart disease and evaluate the performance of multiple classification algorithms on a highly imbalanced healthcare dataset.

The project includes:

* Exploratory Data Analysis (EDA)
* Data Cleaning and Feature Engineering
* Handling Class Imbalance
* Feature Selection and Transformation
* Machine Learning Model Development
* Model Evaluation and Comparison
* Clinical Risk Factor Interpretation
* Data Visualization and Reporting

---

## Dataset Information

* Total Records: **62,001**
* Training Samples: **43,400**
* Test Samples: **18,601**
* Clinical Variables: **11**
* Heart Disease Cases: **~4.75%**

The dataset contains demographic information, lifestyle characteristics, and medical indicators associated with cardiovascular disease risk.

---

## Objectives

* Predict the likelihood of heart disease occurrence.
* Identify key clinical and demographic risk factors.
* Compare multiple machine learning algorithms.
* Address severe class imbalance in healthcare data.
* Generate interpretable insights for healthcare decision-making.

---

## Methodology

### 1. Data Preprocessing

* Missing value treatment
* Outlier detection and removal
* Feature engineering
* Data transformation and scaling
* Categorical variable encoding

---

### 2. Exploratory Data Analysis (EDA)

Performed extensive analysis to understand relationships between risk factors and heart disease.

#### Key Findings

| Risk Factor    | Observation                                           |
| -------------- | ----------------------------------------------------- |
| Age            | Heart disease patients were significantly older       |
| Glucose Level  | Higher glucose levels increased disease risk          |
| BMI            | Elevated BMI showed positive association with disease |
| Hypertension   | Strong predictor of heart disease                     |
| Smoking Status | Former smokers demonstrated higher disease prevalence |

---

### 3. Feature Engineering

Implemented several preprocessing techniques:

* Removal of data leakage variables
* Missing value imputation
* Outlier treatment using IQR methodology
* Log transformation of glucose levels
* One-hot encoding for categorical features
* Standardization using StandardScaler

---

### 4. Machine Learning Models

Developed and compared six classification algorithms:

| Model                        |
| ---------------------------- |
| Logistic Regression          |
| Decision Tree                |
| Random Forest                |
| Support Vector Machine (SVM) |
| AdaBoost                     |
| Gradient Boosting            |

---

### 5. Model Performance Evaluation

Models were evaluated using:

* ROC-AUC Score
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Precision-Recall Curves

#### Performance Summary

| Model               | ROC-AUC    |
| ------------------- | ---------- |
| Gradient Boosting   | **0.8753** |
| Random Forest       | 0.8708     |
| AdaBoost            | 0.8699     |
| Logistic Regression | 0.8678     |
| Decision Tree       | 0.8577     |
| SVM                 | 0.6773     |

#### Best Performing Model

**Gradient Boosting Classifier**

* ROC-AUC: **0.8753**
* Recall: **87.8%**
* Precision: **14.6%**
* F1 Score: **25.1%**

The Gradient Boosting model achieved the highest predictive performance and demonstrated strong capability in identifying high-risk patients.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* scipy
* openpyxl

---

## Project Structure

```text
heart-disease-risk-prediction/
│
├── data/
│   ├── train.xlsx
│   └── test.xlsx
│
├── notebooks/code/
│   └── Heart_Disease_EDA_ML.ipynb
│
├── src/
│   └── heart_disease_analysis.py
│
├── outputs/
│   ├── plots/
│   ├── reports/
│   └── model_results/
│
├── README.md
└── LICENSE
```

---

## Key Insights

* Age emerged as the strongest predictor of heart disease risk.
* Elevated glucose levels and hypertension significantly increased disease probability.
* Former smokers demonstrated higher cardiovascular risk compared to non-smokers.
* Gradient Boosting provided the best overall predictive performance.
* Threshold optimization improved detection of minority-class heart disease cases.

---

## Sample Outputs

### Exploratory Visualizations

* Target Class Distribution
* Age Distribution Analysis
* BMI Distribution
* Glucose Level Analysis
* Correlation Heatmap

### Model Evaluation Visualizations

* ROC Curves
* Precision-Recall Curves
* Confusion Matrices
* Feature Importance Rankings
* Model Performance Comparison Charts

### Prediction Outputs

* Individual Risk Predictions
* Probability Scores
* Model Comparison Metrics

---

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/heart-disease-risk-prediction.git
```

### Navigate to Project Directory

```bash
cd heart-disease-risk-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/Heart_Disease_EDA_ML.ipynb
```

OR execute the complete pipeline script:

```bash
python src/heart_disease_analysis.py
```

---

## Applications

This project demonstrates practical applications of:

* Predictive Healthcare Analytics
* Clinical Risk Assessment
* Machine Learning in Healthcare
* Cardiovascular Disease Prediction
* Data Science for Public Health
* Medical Decision Support Systems

---

## Conclusion

* Gradient Boosting achieved the highest ROC-AUC score (**0.8753**).
* Age, glucose level, BMI, hypertension, and smoking history were identified as major risk factors.
* The project successfully addressed severe class imbalance and maintained high recall for disease detection.
* Results demonstrate the potential of machine learning techniques for early cardiovascular risk assessment.

---

## Future Improvements

* Implement XGBoost and LightGBM models
* Apply SMOTE-based oversampling techniques
* Hyperparameter optimization using GridSearchCV
* Develop a web-based risk prediction dashboard
* Integrate explainable AI techniques (SHAP/LIME)
* Validate model performance on external healthcare datasets
