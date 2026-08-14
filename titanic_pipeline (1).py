from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# TITANIC ANALYTICS MODULE
# ============================================================

BASE_DIR = Path(_file_).resolve().parent

# Find Titanic training data
possible_files = [
    BASE_DIR / "train_data.csv",
    BASE_DIR / "titanic.csv",
    BASE_DIR / "train.csv",
    BASE_DIR / "data_pipeline" / "train_data.csv",
]

data_file = None

for file in possible_files:
    if file.exists():
        data_file = file
        break

if data_file is None:
    raise FileNotFoundError(
        "Titanic training data not found. "
        "Please keep train_data.csv in the repository root."
    )

df = pd.read_csv(data_file)

print("=" * 60)
print("TITANIC ANALYTICS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Target column
# ------------------------------------------------------------

target_candidates = ["Survived", "survived", "SURVIVED"]

target = None
for col in target_candidates:
    if col in df.columns:
        target = col
        break

if target is None:
    raise ValueError("Target column 'Survived' was not found.")

# ------------------------------------------------------------
# Basic analysis
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df[target].value_counts())

survival_rate = df[target].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")

# ------------------------------------------------------------
# Simple Titanic insights
# ------------------------------------------------------------

if "Sex" in df.columns:
    print("\nSurvival rate by Sex:")
    print(df.groupby("Sex")[target].mean())

if "Pclass" in df.columns:
    print("\nSurvival rate by Passenger Class:")
    print(df.groupby("Pclass")[target].mean())

# ------------------------------------------------------------
# Remove unnecessary columns
# ------------------------------------------------------------

drop_columns = [target]

for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=drop_columns)
y = df[target]

# ------------------------------------------------------------
# Identify numeric and categorical columns
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# ------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),
}

results = []
fitted_models = {}

# ------------------------------------------------------------
# Train and evaluate
# ------------------------------------------------------------

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = np.nan

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC AUC": auc,
        }
    )

    fitted_models[model_name] = pipeline

# ------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="ROC AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

# Save model comparison
results_df.to_csv(
    BASE_DIR / "titanic_model_comparison.csv",
    index=False
)

# ------------------------------------------------------------
# Select best model
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_models[best_model_name]

print("\nBest model:", best_model_name)

# ------------------------------------------------------------
# Save complete fitted pipeline
# ------------------------------------------------------------

pipeline_file = BASE_DIR / "titanic_pipeline.joblib"

joblib.dump(best_pipeline, pipeline_file)

print("Saved pipeline:", pipeline_file)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

best_predictions = best_pipeline.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "titanic_confusion_matrix.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# Survival analysis chart
# ------------------------------------------------------------

if "Sex" in df.columns:

    survival_by_sex = df.groupby("Sex")[target].mean()

    plt.figure(figsize=(6, 5))
    survival_by_sex.plot(kind="bar")

    plt.title("Titanic Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "titanic_survival_by_sex.png",
        dpi=150
    )

    plt.close()

# ------------------------------------------------------------
# Passenger class chart
# ------------------------------------------------------------

if "Pclass" in df.columns:

    survival_by_class = df.groupby("Pclass")[target].mean()

    plt.figure(figsize=(6, 5))
    survival_by_class.plot(kind="bar")

    plt.title("Titanic Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "titanic_survival_by_class.png",
        dpi=150
    )

    plt.close()

# ------------------------------------------------------------
# Final recommendation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RECOMMENDATION")
print("=" * 60)

print(
    f"The recommended Titanic prediction model is "
    f"{best_model_name}, selected using ROC AUC."
)

print(
    "The analysis shows that passenger characteristics such as "
    "sex, passenger class and age-related information are useful "
    "for predicting survival."
)

print(
    "The complete preprocessing and prediction pipeline has been "
    "saved as titanic_pipeline.joblib and can be reused on new data."
)

print("\nTitanic analytics completed successfully.")
