# LSTM-Based Context-Aware Anomaly Detection in Time Series data with Dynamic threshold using deep learning
## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Tools & Technologies](#tools--technologies)
5. [Methodology](#methodology)
   - [Data Preprocessing](#1-data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [Sequence Generation for LSTM](#3-sequence-generation-for-lstm)
   - [Model Architecture](#4-model-architecture)
   - [Threshold Optimization](#5-threshold-optimization)
   - [Evaluation](#6-evaluation)
6. [Results & Performance Analysis](#results--performance-analysis)
7. [Reproducibility](#reproducibility)
8. [Future Work & Limitations](#future-work--limitations)
9. [Author](#author)
## Project Overview
This project implements a **context-aware anomaly detection framework** for time series data using a hybrid deep learning approach. The system combines **LSTM networks** with contextual features and dynamic thresholding to detect anomalies in telemetry and sensor data.

---

## Problem Statement
Detecting anomalies in time series is critical for applications in **healthcare, finance, and cybersecurity**. Traditional methods struggle with temporal dependencies and context-specific deviations. This project aims to leverage deep learning to **learn normal patterns** and identify anomalies effectively.

---

## Dataset
- **Source:** Synthetic controlled anomalies dataset  
- **Size:** 5 million timestamps (1M normal, 4M anomalies)  
- **Format:** CSV with features and labels (`y`)  
- **Key Features:** Sensor readings (`aimp, amud, adbr, adfl, arnd, asin1, asin2, bed1`) and contextual features (`hour, dayofweek`)
Due to dataset size and preprocessing dependencies, the full training pipeline is not included. The repository provides the core architecture implementation and experimental results.

---

---

## Tools & Technologies
- **Programming Language:** Python 3.x  
- **Libraries:**  
  - Data Handling: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn`  
  - Deep Learning: `tensorflow.keras`  
  - Imbalanced Data Handling: `imblearn`  
  - PDF Report: `reportlab`  
- **Environment:** Google Colab / VS Code  

---

## Methodology
### 1. Data Preprocessing
- Remove duplicates and missing values  
- Convert timestamp to `datetime`  
- Drop constant or irrelevant columns  
- Feature engineering: rolling means, lag/diff features  
- Contextual features: hour, dayofweek with cyclic encoding  

### 2. Exploratory Data Analysis (EDA)
- Visualize anomalies over time  
- Plot distributions of sensor readings  
- Correlation analysis among features  
- Time series plots for key sensors  

### 3. Sequence Generation for LSTM
- Sliding window sequences with context  
- Train/validation/test split **before oversampling**  
- Oversample only the training set to handle class imbalance  

### 4. Model Architecture
- **Hybrid LSTM Model**:  
  - Bidirectional LSTM layers with Dropout & BatchNorm  
  - Context input merged with LSTM outputs  
  - Dense layers and final sigmoid activation for binary classification  
- **Training:** Binary cross-entropy, Adam optimizer  
- **Callbacks:** Early stopping and learning rate reduction  
The repository includes the core architecture implementation in:


lstm-example.py

### 5. Threshold Optimization
- Use validation set to select optimal F1-score threshold  
- Convert predicted probabilities to class labels  

### 6. Evaluation
- Classification reports and confusion matrices for train, validation, and test sets  
- Precision-recall curve  
- ROC curves saved and compiled into a PDF report  

---

## Key Insights
- Context-aware features improve anomaly detection performance  
- Dynamic thresholding balances precision and recall effectively  
- Oversampling helps address class imbalance without leaking validation/test data  

---


## Results & Performance Analysis

### Train Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 0.99 | 1.00 | 1.00 | 269455 |
| 1.0   | 0.99 | 0.97 | 0.98 | 80836 |
| **Accuracy** |  |  | **0.99** | 350291 |
| **Macro Avg** | 0.99 | 0.99 | 0.99 | 350291 |
| **Weighted Avg** | 0.99 | 0.99 | 0.99 | 350291 |


---

### Validation Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.00 | 1.00 | 1.00 | 57741 |
| 1.0   | 0.92 | 0.94 | 0.93 | 2259 |
| **Accuracy** |  |  | **0.99** | 60000 |
| **Macro Avg** | 0.96 | 0.97 | 0.97 | 60000 |
| **Weighted Avg** | 0.99 | 0.99 | 0.99 | 60000 |

---

### Test Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.00 | 1.00 | 1.00 | 57741 |
| 1.0   | 0.91 | 0.95 | 0.93 | 2259 |
| **Accuracy** |  |  | **0.99** | 60000 |
| **Macro Avg** | 0.96 | 0.97 | 0.96 | 60000 |
| **Weighted Avg** | 0.99 | 0.99 | 0.99 | 60000 |

---

### Analytical Insights

- The model achieves ~99% accuracy across all splits.
- High precision and recall for anomaly class demonstrate strong generalization.
- Minor precision drop from training to validation/test suggests controlled regularization.
- No significant overfitting observed.
- Context integration improves anomaly discrimination capability.

---

## Reproducibility

Dataset: Controlled Anomalies Time Series Dataset (Kaggle)

To reproduce experiments:
- Place cleaned dataset inside `/data`
- Run the training pipeline script

Note: Full preprocessing pipeline is omitted due to dataset size and environment dependencies.

---

## Future Work & Limitations

- Current framework uses supervised learning; future work may explore unsupervised and self-supervised anomaly detection.
- Theoretical optimization techniques could replace empirical threshold tuning.
- Validation on real-world industrial datasets would strengthen generalization claims.
- Transformer-based architectures may improve long-range temporal modeling.

---

## Author

**Akaasha Asif**  
Email: akashaasif99@gmail.com  
LinkedIn: https://www.linkedin.com/in/akaasha-asif

---

