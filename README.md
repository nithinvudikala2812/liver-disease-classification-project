# Liver Disease Classification Using Machine Learning

## Project Overview
This project focuses on developing an end-to-end **multi-class machine learning system** to classify liver disease conditions using clinical and biochemical parameters. The model predicts whether a patient falls into one of the following categories:

- No Disease  
- Suspect Disease  
- Hepatitis C  
- Fibrosis  
- Cirrhosis  

The objective is to support **early diagnosis and decision-making** using data-driven techniques.

---

## Business Objective
Liver diseases often show overlapping symptoms across different stages, making early diagnosis challenging.  
This project aims to build an automated classification model that accurately predicts liver disease categories based on laboratory test results.

---

## Dataset Description
- **Total Records (Raw):** 630  
- **Records Used After Cleaning:** 583  
- **Target Variable:** Disease Category (5 classes)  
- **Feature Count:** 12  

### Key Features:
- Age  
- Sex  
- Albumin  
- Alkaline Phosphatase  
- Alanine Aminotransferase (ALT)  
- Aspartate Aminotransferase (AST)  
- Bilirubin  
- Cholinesterase  
- Cholesterol  
- Creatinine  
- Gamma Glutamyl Transferase (GGT)  
- Protein  

---

## Data Preprocessing
The following preprocessing steps were performed:
- Missing value treatment using mean/median imputation  
- Label encoding for categorical variables  
- Outlier detection and removal using IQR method  
- Feature scaling using standardization  
- Train-test split for model evaluation  

---

## Exploratory Data Analysis (EDA)
- Distribution analysis of clinical attributes  
- Outlier visualization using boxplots  
- Correlation analysis to understand relationships between liver enzymes and disease severity  
- Identification of class imbalance in disease categories  

---

## Handling Class Imbalance
The dataset exhibited significant class imbalance.  
To address this:
- **SMOTE (Synthetic Minority Oversampling Technique)** was applied on the training data  
- Model performance was evaluated both **before and after SMOTE** to ensure fair comparison  

---

## Machine Learning Models Implemented
Multiple classification algorithms were trained and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  
- LightGBM  
- Neural Networks (baseline experimentation)  

Model performance was compared using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC  

---

## Best Performing Model
**Random Forest Classifier** demonstrated the most balanced and stable performance:

- Strong overall accuracy  
- Improved recall for minority classes after SMOTE  
- Robust to noise and outliers  
- Consistent cross-validation results  

This model was selected for deployment.

---

## Deployment
A **Streamlit-based web application** was developed to demonstrate real-time predictions.

### Application Features:
- Manual single-record prediction  
- Batch prediction via file upload  
- Display of predicted disease class  

To run the application locally:
```bash
streamlit run Model_Dep_app.py
