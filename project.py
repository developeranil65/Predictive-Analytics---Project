import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the Dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Data Cleaning
df.columns = df.columns.str.strip()  # Remove spaces from column names

# Remove spaces from string columns
cat_cols = ['education', 'self_employed', 'loan_status']
for col in cat_cols:
    df[col] = df[col].str.strip()

# Handling negative values in assets
df['residential_assets_value'] = df['residential_assets_value'].abs()

print("\nDataset Head:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.info())
print(df.describe())

# Loan Status Distribution
print(df['loan_status'].value_counts())

# Visualization 1: Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df, palette='pastel')
plt.title("Count of Approved vs Rejected Loans")
plt.show()

# Visualization 2: Correlation Matrix
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Financial Features")
plt.show()

# Visualization 3: CIBIL Score vs Loan Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='loan_status', y='cibil_score', data=df)
plt.title("CIBIL Score Distribution by Loan Status")
plt.show()

# Separate target and features
target_col = "loan_status"

# Drop 'loan_id' as it is just an identifier and has no predictive value
X = df.drop(columns=[target_col, 'loan_id'])
y = df[target_col]

print("\nShape of feature matrix:", X.shape)

# Encoding Categorical Features
# Using LabelEncoder as the categorical features are binary
le = LabelEncoder()

# Encode Features
X['education'] = le.fit_transform(X['education'])         # Graduate:0, Not Graduate:1
X['self_employed'] = le.fit_transform(X['self_employed']) # No:0, Yes:1

# Encode Target
y = le.fit_transform(y) # Approved:0, Rejected:1

print("\nEncoded Feature Matrix Head:")
print(X.head())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

print("\nTraining set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Feature Scaling
# Using StandardScaler because financial data (Income, Loan Amount) varies significantly in scale compared to CIBIL score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL 1: Logistic Regression

print("\nModel 1: Logistic Regression")
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

conf_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log) * 100

print("Confusion Matrix:\n", conf_log)
print(f"Accuracy: {acc_log:.2f}%")
print(classification_report(y_test, y_pred_log))

# MODEL 2: Decision Tree Classifier

print("\nModel 2: Decision Tree Classifier")
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

conf_tree = confusion_matrix(y_test, y_pred_tree)
acc_tree = accuracy_score(y_test, y_pred_tree) * 100

print("Confusion Matrix:\n", conf_tree)
print(f"Accuracy: {acc_tree:.2f}%")
print(classification_report(y_test, y_pred_tree))

# Feature Importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tree_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Features impacting Loan Approval:")
print(importance_df.head(5))

# Plot the Tree
plt.figure(figsize=(15, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=['Approved', 'Rejected'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# MODEL 3: K-nearest neighbours (KNN)

print("\nModel 3: K-Nearest Neighbors (KNN)")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

conf_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn) * 100

print("Confusion Matrix:\n", conf_knn)
print(f"Accuracy: {acc_knn:.2f}%")
print(classification_report(y_test, y_pred_knn))

# MODEL 4: Naive Bayes

print("\nModel 4: Gaussian Naive Bayes")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

conf_nb = confusion_matrix(y_test, y_pred_nb)
acc_nb = accuracy_score(y_test, y_pred_nb) * 100

print("Confusion Matrix:\n", conf_nb)
print(f"Accuracy: {acc_nb:.2f}%")
print(classification_report(y_test, y_pred_nb))

# MODEL 5: Support Vector Machine (SVM)

print("\nModel 5: Support Vector Machine (SVM)")
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

conf_svm = confusion_matrix(y_test, y_pred_svm)
acc_svm = accuracy_score(y_test, y_pred_svm) * 100

print("Confusion Matrix:\n", conf_svm)
print(f"Accuracy: {acc_svm:.2f}%")
print(classification_report(y_test, y_pred_svm))

# Final Accuracy Summary & Conclusion
accuracy_summary = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ],
    "Accuracy (%)": [
        acc_log,
        acc_tree,
        acc_knn,
        acc_nb,
        acc_svm
    ]
})

print("\nFinal Accuracy Summary:")
print(accuracy_summary.sort_values(by="Accuracy (%)", ascending=False))

plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy (%)", y="Model", data=accuracy_summary, palette="viridis")
plt.title("Model Comparison by Accuracy")
plt.xlim(0, 100)
plt.show()

# Saving Predictions
# Creating a CSV with Actual vs Predicted values for all models
prediction_df = pd.DataFrame({
    "Actual_Status": y_test,
    "LogReg_Pred": y_pred_log,
    "DecisionTree_Pred": y_pred_tree,
    "KNN_Pred": y_pred_knn,
    "NaiveBayes_Pred": y_pred_nb,
    "SVM_Pred": y_pred_svm
})

# Decode the 'Actual_Status' back to "Approved"/"Rejected" for readability
# 0 means Approved and 1 means Rejected based on LabelEncoder
prediction_df['Actual_Status_Label'] = prediction_df['Actual_Status'].apply(lambda x: 'Rejected' if x == 1 else 'Approved')

print("\nSample Predictions (First 10 rows):")
print(prediction_df.head(10))

prediction_df.to_csv("loan_approval_predictions_all_models.csv", index=False)
print("\nPrediction file 'loan_approval_predictions_all_models.csv' saved successfully")