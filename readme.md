# 🏦 Loan Approval Prediction System

## Project Overview
This project implements a **Predictive Analytics Machine Learning system** to automate loan approval decisions for a financial institution.

The model analyzes applicant demographics, financial indicators, and credit history to predict whether a loan application should be **Approved** or **Rejected**.

### Objectives
- Reduce manual decision-making effort
- Minimize default risk
- Improve approval speed and consistency
- Apply **Supervised Learning** concepts in a real-world financial use case

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=developeranil65/Loan-approval-predictive-analytics&type=date&legend=top-left)](https://www.star-history.com/#developeranil65/Loan-approval-predictive-analytics&type=date&legend=top-left)

## 📂 Dataset Description
- **Total Records:** 4,269  
- **Total Features:** 13  
- **Target Variable:** `loan_status` (Approved / Rejected)

### Key Predictors
- `cibil_score`
- `income_annum`
- `loan_amount`
- `loan_term`
- `education`
- `self_employed`

### Asset-Based Features
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`

The dataset reflects real-world financial patterns and is suitable for classification modeling and risk analysis.

---

## Tech Stack
- **Language:** Python  
- **Libraries:**  
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  

### Techniques Used
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Supervised Classification Modeling  
- Model Evaluation & Comparison  

---

## Exploratory Data Analysis (EDA)

### 1. Loan Approval Distribution
Checked class balance to ensure unbiased model training.

**Figure 1:** Approved vs Rejected loan count
![Approved vs Rejected loan count](images/Figure_1.png)

---

### 2. Feature Correlation Analysis
Generated a correlation heatmap to understand relationships between financial attributes.

Key Insight:
- Strong correlation between `loan_amount` and `income_annum`

**Figure 2:** Correlation heatmap of financial features
![Correlation heatmap](images/Figure_2.png)

---

### 3. Impact of CIBIL Score
CIBIL Score emerged as the most influential feature.

Insight:
- Higher CIBIL scores significantly increase approval probability

**Figure 3:** Boxplot comparing CIBIL scores for Approved vs Rejected applicants
![Boxplot](images/Figure_3.png)

---

## Models Implemented
Five supervised classification models were implemented and evaluated:

1. **Logistic Regression**  
   - Baseline probabilistic classifier  

2. **Decision Tree Classifier**  
   - Highly interpretable, rule-based model  

3. **K-Nearest Neighbors (KNN)**  
   - Instance-based learning using similarity  

4. **Gaussian Naive Bayes**  
   - Probabilistic model assuming feature independence  

5. **Support Vector Machine (SVM)**  
   - Finds optimal decision boundary between classes  

---

## Model Performance & Results

Train-Test Split: **80% Training / 20% Testing**

| Rank | Model               | Accuracy | F1 Score |
|-----:|---------------------|----------|----------|
| 1    | Decision Tree       | 97.07%   | 0.96     |
| 2    | Naive Bayes         | 93.68%   | 0.92     |
| 3    | SVM                 | 91.57%   | 0.89     |
| 4    | Logistic Regression | 90.52%   | 0.87     |
| 5    | KNN                 | 89.23%   | 0.86     |

---

### 4. Model Comparison Visualization
**Figure 4:** Accuracy comparison across all 5 models
![Accuracy comparison](images/Figure_5.png)

---

### 5. Decision Tree Visualization
The best-performing model (Decision Tree) was visualized after pruning.

Key Observations:
- **CIBIL Score** is the root node (primary decision factor)
- Followed by **Loan Term** and **Asset Values**

**Figure 5:** Pruned Decision Tree showing approval rules
![Decision Tree](images/Figure_4.png)

---

## Conclusion
The **Decision Tree Classifier** outperformed all other models with ~97% accuracy.

### Why it worked best:
- Loan approval is inherently **rule-based**
- Decision Trees naturally model hierarchical decision logic  
  *(e.g., “If CIBIL > 750 → check income → check assets”)*

### Business Impact
- Transparent & explainable decisions  
- High accuracy with minimal complexity  
- Suitable for real-world banking automation  
- Reduces bad loan risk significantly  

---

## ⚙️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/developeranil65/Loan-approval-predictive-analytics.git
cd loan-approval-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the main script
```bash
python loan_approval_project.py
```

### 4. View the generated CSV `loan_approval_predictions_all_models.csv` for the final predictions


