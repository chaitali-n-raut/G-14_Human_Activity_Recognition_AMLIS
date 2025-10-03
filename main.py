import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import os
import urllib.request
import zipfile
import time
import random

# -----------------------------
# 1. Download & Prepare Dataset
# -----------------------------
def download_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = "UCI_HAR_Dataset.zip"
    extract_path = "UCI HAR Dataset"

    if not os.path.exists(extract_path):
        print("[INFO] Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)

        print("[INFO] Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)
        print("[INFO] Dataset ready!")
    else:
        print("[INFO] Dataset already exists.")

# -----------------------------
# 2. Load Data
# -----------------------------
def load_data():
    feature_names = pd.read_csv("UCI HAR Dataset/features.txt",
                                sep=r"\s+", header=None, usecols=[1])
    feature_names = feature_names[1].values

    # Train data
    X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt",
                          sep=r"\s+", header=None)
    X_train.columns = feature_names
    y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", header=None)[0]

    # Test data
    X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt",
                         sep=r"\s+", header=None)
    X_test.columns = feature_names
    y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", header=None)[0]

    # Activity labels
    activity_labels = pd.read_csv("UCI HAR Dataset/activity_labels.txt",
                                  sep=r"\s+", header=None, index_col=0)[1].to_dict()

    y_train = y_train.map(activity_labels)
    y_test = y_test.map(activity_labels)

    return X_train, X_test, y_train, y_test

# -----------------------------
# 3. Train & Evaluate
# -----------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# -----------------------------
# 4. Dynamic Predictions (Live-style)
# -----------------------------
def dynamic_prediction(model, X_test, y_test, n_samples=5):
    print("\n[INFO] Live Activity Predictions 🔥\n")
    for _ in range(n_samples):
        idx = random.randint(0, len(X_test)-1)
        sample = X_test.iloc[idx:idx+1]
        true_label = y_test.iloc[idx]

        # Prediction
        pred_probs = model.predict_proba(sample)[0]
        top3_idx = np.argsort(pred_probs)[::-1][:3]

        print(f"🎯 Sample #{idx}")
        print(f"   ✅ Actual Activity: {true_label}")
        print("   🤖 Predicted (Top 3):")
        for rank, i in enumerate(top3_idx, start=1):
            bar = "█" * int(pred_probs[i]*50)   # scaled to 50 blocks
            print(f"     {rank}. {model.classes_[i]} "
                  f"(Confidence: {pred_probs[i]*100:.2f}%) {bar}")
        print("-"*60)

        time.sleep(1)  # simulate live stream

# -----------------------------
# 5. Extract Example Values (20 per activity)
# -----------------------------
def get_sample_for_all_activities(X_test, y_test, n_values=20):
    activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                  "SITTING", "STANDING", "LAYING"]

    print("\n👉 Example 20 values for manual input (rounded to 4 decimals):\n")
    for activity in activities:
        sample = X_test[y_test == activity].iloc[0].values
        sample = np.round(sample[:n_values], 4)
        print(f"{activity}:")
        print(sample.tolist())
        print()

# -----------------------------
# 6. Main Execution
# -----------------------------
if __name__ == "__main__":
    download_dataset()
    print("[INFO] Loading UCI HAR Dataset...")
    X_train, X_test, y_train, y_test = load_data()

    print("[INFO] Training Models...\n")

    # Train models
    dt = evaluate_model(DecisionTreeClassifier(random_state=42), X_train, X_test, y_train, y_test, "Decision Tree")
    knn = evaluate_model(KNeighborsClassifier(n_neighbors=5), X_train, X_test, y_train, y_test, "KNN")
    svm = evaluate_model(SVC(kernel="linear", probability=True, random_state=42),
                         X_train, X_test, y_train, y_test, "SVM")

    # Save best model (SVM)
    model_path = "HAR_New/models/svm_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(svm, model_path)

    # ✅ Save mean feature values (for manual input)
    np.save("HAR_New/models/train_mean.npy", X_train.mean(axis=0))

    print(f"[INFO] SVM model and means saved at: {model_path}")

    # Run dynamic predictions using best model (SVM)
    print("\n[INFO] Running dynamic predictions with SVM model ⚡\n")
    dynamic_prediction(svm, X_test, y_test, n_samples=5)

    # Print 20 values for all activities
    get_sample_for_all_activities(X_test, y_test, n_values=20)
