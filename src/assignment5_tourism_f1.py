import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)


# -------------------------
# Helper: Save Confusion Matrix
# -------------------------
def save_confusion_matrix(cm, path, title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_normalized(cm, path):
    cm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, where=row_sum != 0) * 100

    plt.figure()
    plt.imshow(cm_norm)
    plt.title("Confusion Matrix (Normalized %)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.1f}", ha="center", va="center")

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Find Best Threshold (Maximize F1 for class 1)
# -------------------------
def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = 0

    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def main():

    DATA_PATH = "data/tourism_clean.csv"
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -------------------------
    # Load Data
    # -------------------------
    df = pd.read_csv(DATA_PATH)

    if "ProdTaken" not in df.columns:
        raise ValueError("ProdTaken column not found.")

    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"].astype(int)

    # Detect categorical & numeric columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # -------------------------
    # Train/Test Split
    # -------------------------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    # Validation split (for threshold tuning)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=42
    )

    # -------------------------
    # Model Candidates
    # -------------------------
    models = {
        "LogisticRegression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                C=1.0
            ))
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    best_model = None
    best_name = None
    best_threshold = 0.5
    best_val_f1 = 0

    # -------------------------
    # Train + Threshold Tune
    # -------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        threshold, f1_val = find_best_threshold(y_val, y_val_prob)

        print(f"{name} - Best VAL F1: {f1_val:.4f} at threshold {threshold:.2f}")

        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            best_model = model
            best_name = name
            best_threshold = threshold

    print("\nBest Model:", best_name)
    print("Best Threshold:", best_threshold)

    # -------------------------
    # Retrain on Full Train Data
    # -------------------------
    best_model.fit(X_train_full, y_train_full)

    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_prob >= best_threshold).astype(int)

    # -------------------------
    # Evaluation
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== FINAL TEST RESULTS ===")
    print("Model:", best_name)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (1): {prec:.4f}")
    print(f"Recall (1): {rec:.4f}")
    print(f"F1-score (1): {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # -------------------------
    # Save Files
    # -------------------------
    with open(os.path.join(RESULTS_DIR, "tourism_evaluation.txt"), "w") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"Threshold: {best_threshold:.2f}\n\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Precision (1): {prec:.6f}\n")
        f.write(f"Recall (1): {rec:.6f}\n")
        f.write(f"F1-score (1): {f1:.6f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    save_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    save_confusion_matrix_normalized(cm, os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png"))

    preds = pd.DataFrame({
        "y_true": y_test.values,
        "y_prob": y_test_prob,
        "y_pred": y_pred
    })
    preds.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

    print("\nAll results saved in 'results/' folder.")


if __name__ == "__main__":
    main()
