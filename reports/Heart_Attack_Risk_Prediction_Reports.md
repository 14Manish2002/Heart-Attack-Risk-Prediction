# Heart Disease Risk Prediction — Project Report

## 1. Problem Statement

Heart disease is one of the leading causes of death worldwide. Early identification of at-risk patients enables timely intervention and can save lives. This project builds a machine learning pipeline to predict whether a patient has heart disease based on clinical and demographic features.

**Target Variable**: `heart_disease` (0 = No, 1 = Yes)  
**Task**: Binary classification with severe class imbalance (~4.75% positive rate)

---

## 2. Dataset Overview

| Attribute | Train | Test |
|---|---|---|
| Rows | 43,400 | 18,601 |
| Features | 12 | 11 |
| Positive cases | 2,062 (4.75%) | 894 (4.81%) |
| Missing: BMI | 1,462 (3.4%) | 591 (3.2%) |
| Missing: Smoking | 13,292 (30.6%) | 5,751 (30.9%) |

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Target Distribution
The dataset is highly imbalanced: 95.25% of patients do not have heart disease, while only 4.75% do. This class imbalance is a central challenge for model building.

### 3.2 Numerical Features

**Age**
- Strong positive correlation with heart disease (r = 0.25)
- Patients with heart disease: median age ~67
- Patients without heart disease: median age ~41
- Age ranges: 0.08 to 82 years

**Average Glucose Level**
- Moderate positive correlation (r = 0.15)
- Right-skewed distribution (skew = 0.91) → log-transformed
- After outlier removal: range 55–162 mg/dL

**BMI**
- Weak correlation with heart disease (r = 0.05)
- 1,462 missing values (3.4%) — filled with median (27.7)
- 1,778 outliers removed using 1.2×IQR rule

### 3.3 Categorical Features

| Feature | Finding |
|---|---|
| Gender | Males have ~6.5% heart disease rate vs ~3.3% for females |
| Ever Married | Married patients show higher rate (6.5% vs 1.1%) — age-confounded |
| Work Type | Self-employed and Govt job workers have higher rates |
| Smoking Status | Formerly smoked: 8.7%, Smokes: 6.6%, Never smoked: 3.8% |
| Residence Type | No significant difference (chi² p = 0.65) → **dropped** |

### 3.4 Statistical Tests
Chi-square tests showed strong associations between heart disease and:
- `ever_married`: chi² = 721.4, p < 10⁻¹⁵⁹ (strongest)
- `work_type`: chi² = 667.4, p < 10⁻¹⁴³
- `smoking_status`: chi² = 421.9, p < 10⁻⁹¹
- `gender`: chi² = 284.6, p < 10⁻⁶²
- `Residence_type`: chi² = 0.20, p = 0.65 → **not significant**

---

## 4. Data Preprocessing

### 4.1 Feature Engineering
- **Dropped**: `id` (identifier), `stroke` (data leakage), `Residence_type` (not significant)
- **Removed**: 11 rows with gender = "Other" (negligible sample size)
- **Log-glucose**: `log_glucose = log1p(avg_glucose_level)` reduces right skew from 0.91 to 0.41

### 4.2 Missing Value Handling
- **BMI**: Fill with median (27.7) after numeric coercion
- **Smoking status**: Fill with `'Unknown'` category (preserves information)

### 4.3 Outlier Treatment
- **BMI**: Remove extreme values (1.2×IQR rule) — 1,778 rows removed
- **Avg glucose**: Winsorize with 1.5×IQR (cap, do not remove)

### 4.4 Encoding & Scaling
- One-hot encoding with `drop_first=True` (avoids multicollinearity)
- StandardScaler applied to all features
- Final feature matrix: 13 columns

### 4.5 Threshold Tuning
Standard threshold of 0.50 produces near-zero recall on the minority class.  
Domain-tuned threshold: **0.0475** (≈ class prevalence).  
This dramatically improves recall (0.87–0.90) at the cost of precision.

---

## 5. Model Building

Six classification models were trained on the preprocessed training data and evaluated on the test set:

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | max_iter=1000, solver='lbfgs' |
| Decision Tree | max_depth=6 |
| Random Forest | n_estimators=200, max_depth=8 |
| Gradient Boosting | n_estimators=200, lr=0.05, max_depth=4 |
| SVM | kernel='rbf', C=1.0, probability=True |
| AdaBoost | n_estimators=200, learning_rate=0.5 |

---

## 6. Results & Model Comparison

### 6.1 ROC AUC (Primary Metric — threshold-independent)

| Rank | Model | ROC AUC |
|---|---|---|
| 1 ⭐ | Gradient Boosting | **0.8753** |
| 2 | Random Forest | 0.8708 |
| 3 | AdaBoost | 0.8699 |
| 4 | Logistic Regression | 0.8678 |
| 5 | Decision Tree | 0.8577 |
| 6 | SVM | 0.6773 |

### 6.2 Detailed Metrics at Threshold = 0.0475

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Gradient Boosting | 0.748 | 0.146 | **0.878** | **0.251** |
| Logistic Regression | 0.746 | 0.145 | 0.874 | 0.249 |
| Decision Tree | 0.744 | 0.142 | 0.856 | 0.243 |
| Random Forest | 0.725 | 0.138 | **0.899** | 0.239 |

> **Note**: Low precision is expected with 4.75% base rate and high recall target. In medical screening, recall (sensitivity) is prioritised — missing a true case is more costly than a false alarm.

---

## 7. Feature Importance Analysis

### Gradient Boosting Top Features
1. **Age** (0.655) — most important predictor
2. **Log glucose** (0.111) — elevated blood sugar indicates risk
3. **Gender_Male** (0.101) — males at higher risk
4. **BMI** (0.080) — obesity link to heart disease
5. **Hypertension** (0.019) — known co-morbidity

### Random Forest Top Features
1. BMI (0.335)
2. Age (0.282)
3. Log glucose (0.259)
4. Gender_Male (0.024)
5. Hypertension (0.019)

### Logistic Regression Coefficients
1. Age (1.75) — highest positive effect
2. Gender_Male (0.39) — males higher risk
3. Log glucose (0.21)
4. Work_type_children (-0.17) — protective
5. Smoking_smokes (0.16)

---

## 8. Conclusion

### Key Findings
1. **Best model**: Gradient Boosting with ROC AUC = 0.8753
2. **Most critical features**: Age, glucose level, BMI, gender, and hypertension
3. **Threshold tuning** is essential for imbalanced medical datasets — default 0.5 misses 99% of disease cases
4. **SVM** performed poorly (AUC 0.677) — likely due to scaling issues with non-linear boundaries at low thresholds
5. **Random Forest** achieved highest recall (89.9%) — useful for maximum sensitivity screening

### Clinical Implications
- The model can identify ~87–90% of heart disease patients at the cost of ~86–90% false positive rate
- Suitable as a **first-pass screening tool** to flag high-risk patients for further examination
- Age is the dominant risk factor, followed by blood glucose and BMI

### Limitations
- Heavy class imbalance not addressed with SMOTE/undersampling — future improvement
- Smoking status missing for 30% of training data — filled with "Unknown"
- Glucose outliers winsorized rather than removed — may affect model calibration

### Future Work
- Apply SMOTE or class_weight='balanced' to improve minority class detection
- Hyperparameter tuning via GridSearchCV / Optuna
- XGBoost and LightGBM for potential improvement
- Calibration curves to assess probability reliability
- Cross-validation (5-fold stratified) for more robust estimates

---

*Report generated automatically by the ML pipeline*  
*Project: Heart Disease Risk Prediction*
