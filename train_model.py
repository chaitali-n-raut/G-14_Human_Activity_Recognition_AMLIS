import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

# -----------------------------
# Create models folder
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load UCI HAR Dataset
# -----------------------------
X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep=r"\s+", header=None)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", header=None)[0]

X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep=r"\s+", header=None)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", header=None)[0]

# -----------------------------
# Map integers to activity names
# -----------------------------
activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}
y_train_names = y_train.map(activity_map)
y_test_names = y_test.map(activity_map)

# -----------------------------
# Encode labels
# -----------------------------
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train_names)
y_test_enc = encoder.transform(y_test_names)

# -----------------------------
# Select top features (ANOVA F-test)
# -----------------------------
k = 20  # keep small for manual input
selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train_enc)
X_test_selected = selector.transform(X_test)

# Save selected feature indices
selected_indices = selector.get_support(indices=True)
np.save("models/selected_features.npy", selected_indices)
print(f"✅ Saved top {k} selected feature indices for manual input!")

# -----------------------------
# Train SVM
# -----------------------------
model_svm = SVC(kernel="linear", probability=True, random_state=42)
model_svm.fit(X_train_selected, y_train_enc)
joblib.dump(model_svm, "models/svm_model_selected.pkl")
print("✅ SVM trained and saved!")

# -----------------------------
# Train Random Forest
# -----------------------------
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_selected, y_train_enc)
joblib.dump(model_rf, "models/rf_model_selected.pkl")
print("✅ Random Forest trained and saved!")

# Save RF feature importance for visualization
rf_importance = model_rf.feature_importances_
np.save("models/rf_feature_importance.npy", rf_importance)

# -----------------------------
# Train KNN
# -----------------------------
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_selected, y_train_enc)
joblib.dump(model_knn, "models/knn_model_selected.pkl")
print("✅ KNN trained and saved!")

# -----------------------------
# Save encoder and training mean
# -----------------------------
joblib.dump(encoder, "models/label_encoder.pkl")
train_mean = X_train.mean(axis=0).values
np.save("models/train_mean.npy", train_mean)
print("✅ Label encoder and training mean saved!")
