"""
PCOS Risk Prediction Using Machine Learning

Audience-ready version with descriptive graphs and textual explanations.
Includes main model comparison line graph and combined summary figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import shap

# Optional XGBoost support
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# 2. Load and Clean Data
file_path = "PCOS_extended_dataset.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
for col in ["Sl. No", "Patient File No."]:
    if col in df.columns:
        df = df.drop(columns=[col])

TARGET_COLUMN = "PCOS (Y/N)"
if df[TARGET_COLUMN].dtype == object:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip().map({"Y": 1, "N": 0})
else:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({1:1, 0:0})

yn_columns = [
    "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)",
    "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)",
    "Fast food (Y/N)", "Reg.Exercise(Y/N)"
]

for col in yn_columns:
    if col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().map({"Y": 1, "N": 0})
        else:
            df[col] = df[col].map({1:1, 0:0})

for col in df.columns:
    if col != TARGET_COLUMN:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET_COLUMN])
numeric_cols = df.columns.drop(TARGET_COLUMN)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Model Training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)

results = {}
pred_probas = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    }

    pred_probas[name] = y_proba

results_df = pd.DataFrame(results).T.round(3)


# 4. Big Audience-ready Line Graph (Model Comparison)
plt.figure(figsize=(14,6))
for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    plt.plot(results_df.index, results_df[metric], marker='o', markersize=12, linewidth=3, label=metric)

plt.title("Comparison of Machine Learning Models for PCOS Prediction", fontsize=18)
plt.ylabel("Score (0-1)", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.ylim(results_df.min().min() - 0.01, 1.0)  # zoom to see differences
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# 5. Feature Importance (Random Forest)
rf_model = models["Random Forest"]
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)


# 6. SHAP Explainability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)


# 7. Combined Summary Figure (Compact & Readable)
fig, axes = plt.subplots(2, 2, figsize=(12,8))  
plt.subplots_adjust(hspace=0.4, wspace=0.3)    

# 7a. Histogram of predicted probabilities
y_pred_proba = pred_probas["Random Forest"]
axes[0,0].hist(y_pred_proba[y_test==0], bins=20, alpha=0.7, label="No PCOS", color="skyblue")
axes[0,0].hist(y_pred_proba[y_test==1], bins=20, alpha=0.7, label="PCOS", color="salmon")
axes[0,0].set_title("Predicted PCOS Risk Distribution", fontsize=14)
axes[0,0].set_xlabel("Predicted Probability", fontsize=11)
axes[0,0].set_ylabel("Number of Patients", fontsize=11)
axes[0,0].legend(fontsize=10)

# 7b. Scatter of individual predictions
axes[0,1].scatter(range(len(y_test)), y_pred_proba, c=y_test, cmap="bwr", alpha=0.7)
axes[0,1].axhline(0.5, color="green", linestyle="--", label="Threshold 0.5")
axes[0,1].set_title("Predicted Probability per Patient", fontsize=14)
axes[0,1].set_xlabel("Patient Index", fontsize=11)
axes[0,1].set_ylabel("Predicted Probability", fontsize=11)
axes[0,1].legend(fontsize=10)
axes[0,1].set_ylim(0,1)

# 7c. ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1,0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.2f})')
axes[1,0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label="Random Guess")
axes[1,0].set_title("ROC Curve (Random Forest)", fontsize=14)
axes[1,0].set_xlabel("False Positive Rate", fontsize=11)
axes[1,0].set_ylabel("True Positive Rate", fontsize=11)
axes[1,0].legend(fontsize=10)
axes[1,0].grid(True, linestyle='--', alpha=0.5)

# 7d. Top 5 feature importance
sns.barplot(data=importance_df.head(5), x="Importance", y="Feature", palette="viridis", ax=axes[1,1])
axes[1,1].set_title("Top 5 Most Influential Features", fontsize=14)
axes[1,1].set_xlabel("Importance Score", fontsize=11)
axes[1,1].set_ylabel("Feature", fontsize=11)

plt.tight_layout()
plt.show()


# 8. Text Output Descriptions
print("\n=== Graph Explanations ===\n")

print("1. Model Comparison Line Graph:")
print("   - Shows how each model performs on Accuracy, Precision, Recall, and F1 Score.")
print("   - Zoomed to actual range to highlight differences between models.")
print("   - Helps the audience see which model is best overall.\n")

print("2. Predicted PCOS Risk Distribution (Histogram):")
print("   - Red = patients with actual PCOS, Blue = patients without PCOS.")
print("   - Shows how the Random Forest model separates high vs low risk.\n")

print("3. Predicted Probability per Patient (Scatter Plot):")
print("   - Each point is a patient, colored by actual label (0=No, 1=PCOS).")
print("   - Green line shows decision threshold at 0.5.\n")

print("4. ROC Curve:")
print("   - Shows model's ability to discriminate PCOS vs non-PCOS.")
print("   - AUC closer to 1 = better model.\n")

print("5. Top 5 Most Influential Features (Bar Chart):")
print("   - Features that contribute most to predictions, like BMI or hormone ratios.")
print("   - Explains why the model predicts high risk for some patients.\n")


# 9. Example Prediction & Conclusion
example = X_test.iloc[[0]]
pcos_probability = rf_model.predict_proba(example)[0][1]

print("\n=== Example Prediction ===")
print(f"Predicted probability of PCOS (Random Forest): {pcos_probability:.3f}")

best_model = results_df["F1 Score"].idxmax()
top_features = importance_df.head(5)["Feature"].tolist()

print("\n=== Final Conclusions ===")
print(f"Best-performing model: {best_model}")
print("Most influential biological features:")
for feature in top_features:
    print(f"- {feature}")
