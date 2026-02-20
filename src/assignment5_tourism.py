import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt


def save_confusion_matrix(cm, path, title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    # write values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    # -------------------------
    # 0) Paths
    # -------------------------
    DATA_PATH = "data/tourism_clean.csv"
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -------------------------
    # 1) Load data + basic checks (for report)
    # -------------------------
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    if "ProdTaken" not in df.columns:
        raise ValueError("Column 'ProdTaken' not found. Check your CSV header.")

    # dataset overview info
    shape = df.shape
    missing_total = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    class_counts = df["ProdTaken"].value_counts().sort_index()
    class_percent = (class_counts / class_counts.sum() * 100).round(2)

    print("Data shape:", shape)
    print("Total missing values:", missing_total)
    print("Duplicate rows:", duplicates)
    print("\nClass distribution (ProdTaken):")
    print(class_counts)
    print("\nClass distribution (%):")
    print(class_percent)

    # Save dataset overview to file (easy to screenshot / attach)
    overview_path = os.path.join(RESULTS_DIR, "dataset_overview.txt")
    with open(overview_path, "w") as f:
        f.write(f"Data shape: {shape}\n")
        f.write(f"Total missing values: {missing_total}\n")
        f.write(f"Duplicate rows: {duplicates}\n\n")
        f.write("Class distribution (ProdTaken):\n")
        f.write(class_counts.to_string() + "\n\n")
        f.write("Class distribution (%):\n")
        f.write(class_percent.to_string() + "\n")

    print(f"\nSaved dataset overview to: {overview_path}")

    # -------------------------
    # 2) Split X / y
    # -------------------------
    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    # -------------------------
    # 3) Train/Test split (75/25) - stratified
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    print("\nSplit sizes:")
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # -------------------------
    # 4) One-hot encode categorical columns
    # -------------------------
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Ensure test has same columns as train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print("After one-hot encoding:")
    print("Train features:", X_train.shape[1], "Test features:", X_test.shape[1])

    # -------------------------
    # 5) Scale features
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # 6) Train model
    # -------------------------
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # -------------------------
    # 7) Predict + evaluate
    # -------------------------
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report_text)

    # Save evaluation text
    eval_path = os.path.join(RESULTS_DIR, "tourism_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)

    # Save classification report separately (clean)
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nSaved evaluation to: {eval_path}")
    print(f"Saved classification report to: {report_path}")

    # -------------------------
    # 8) Save confusion matrix images
    # -------------------------
    cm_img_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    save_confusion_matrix(cm, cm_img_path, title="Confusion Matrix")
    print(f"Saved confusion matrix image to: {cm_img_path}")

    # Normalized confusion matrix (optional but nice)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = (cm_norm * 100).round(1)  # percent
    cm_norm_img_path = os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png")
    save_confusion_matrix(cm_norm, cm_norm_img_path, title="Confusion Matrix (Normalized %)")
    print(f"Saved normalized confusion matrix image to: {cm_norm_img_path}")

    # -------------------------
    # 9) Save predictions CSV
    # -------------------------
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })
    preds_path = os.path.join(RESULTS_DIR, "predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()
