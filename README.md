# 📊 Financial Sector Classification from Pre/Post-COVID Indicators

This project analyzes and classifies companies into economic sectors based on financial indicators collected **before and after the COVID-19 pandemic**. The goal is to:

- Explore indicator shifts across sectors
- Identify significant changes via statistical testing (ANOVA)
- Train machine learning models to classify companies by sector
- Visualize sector-wise indicator shifts and PCA mappings

---

## 📁 Dataset

The dataset contains multiple `.arff` files, each representing financial data for different quarters and countries. Additionally:

- `dimension.csv`: Describes financial indicators (e.g., X1 → Total assets)
- `sector_dimension.csv`: Maps sector codes (`S`) to their names

---

## 📌 Tasks Overview

### ✅ Task 1: Data Preparation

- Read and parse `.arff` files
- Combine all files into a single DataFrame
- Mark each row as `pre-COVID` (≤ Q1 2020) or `post-COVID` (≥ Q2 2020)
- Validate allowed values from attribute metadata

### 📊 Task 2: Visualization & EDA

- Use heatmaps, PCA, and boxplots to analyze patterns and distributions
- Plot sector distributions and feature relationships
- Visualize significant changes using ANOVA results

### 🧪 Task 3: ANOVA Analysis

- Run one-way ANOVA for each indicator
- Identify features with significant changes (p < 0.05)
- Analyze sector-specific changes in financial indicators

### 🤖 Task 4: Machine Learning

- Use `RandomForestClassifier` to predict company sectors (`S`)
- Perform cross-validation with `f1_macro` scoring
- Tune hyperparameters using `GridSearchCV`
- Evaluate using classification report on held-out test set

---

## 🧠 Insights

- Identified sector-specific features significantly affected by COVID-19
- Top indicators: Market Cap/EBITDA, Working Capital, Cash Flow ratios, etc.
- Achieved strong classification performance (e.g., F1 macro ≈ 0.77)

---

## 🛠️ Tech Stack

- Python 3.9+
- pandas, numpy
- scikit-learn
- scipy.stats (ANOVA)
- seaborn, matplotlib
- `scipy.io.arff` for ARFF file parsing

---

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/financial-sector-prediction.git
   cd financial-sector-prediction
   pip install -r requirements.txt
   python3 main.py
