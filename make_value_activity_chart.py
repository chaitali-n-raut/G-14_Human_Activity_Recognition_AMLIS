import pandas as pd
import numpy as np
import joblib

# Load trained model and encoder
model = joblib.load("models/svm_model_selected.pkl")
encoder = joblib.load("models/label_encoder.pkl")
selected_features = np.load("models/selected_features.npy")

# Load test dataset
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None)

# Pick first 10 samples for demo (to keep chart small)
X_test_selected = X_test.iloc[:10, selected_features]

# Predict activities
preds = model.predict(X_test_selected)
pred_labels = encoder.inverse_transform(preds)

# Build result DataFrame
result_df = X_test_selected.copy()
result_df["Predicted Activity"] = pred_labels

# Replace feature column names with friendly labels if available
feature_name_map = {
    2: "📈 Accelerometer Mean (Z)",
    9: "Other Feature (ID 9)",
    15: "Other Feature (ID 15)",
    40: "Other Feature (ID 40)",
    49: "Other Feature (ID 49)",
    52: "Other Feature (ID 52)",
    56: "Other Feature (ID 56)",
    102: "Other Feature (ID 102)",
    103: "Other Feature (ID 103)",
    104: "Other Feature (ID 104)",
    234: "Other Feature (ID 234)",
    268: "Other Feature (ID 268)",
    271: "Other Feature (ID 271)",
    280: "Other Feature (ID 280)",
    287: "Other Feature (ID 287)",
    366: "Other Feature (ID 366)",
    367: "Other Feature (ID 367)",
    368: "Other Feature (ID 368)",
    523: "Other Feature (ID 523)",
    558: "Other Feature (ID 558)"
}
result_df = result_df.rename(columns=feature_name_map)

# Save chart
result_df.to_csv("values_activity_chart.csv", index=False)
print("✅ Chart saved as values_activity_chart.csv")
print(result_df.head(5))
