import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import base64
import re

# -----------------------------
# Paths
# -----------------------------
MODEL_SVM_PATH = "models/svm_model_selected.pkl"
MODEL_RF_PATH = "models/rf_model_selected.pkl"
MODEL_KNN_PATH = "models/knn_model_selected.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURES_PATH = "models/selected_features.npy"
MEAN_PATH = "models/train_mean.npy"
CHART_PATH = "values_activity_chart.csv"
FEATURES_TXT_PATH = "UCI HAR Dataset/features.txt"
BANNER_IMAGE_PATH = r"E:\B.Tech\SEM5\ML\HAR_New\ban.png"   # update if needed

# -----------------------------
# Utilities
# -----------------------------
def safe_joblib_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load {path}: {e}")
            return None
    return None

def readable_name(raw_name: str) -> str:
    """Convert raw feature name to a more human-readable variant."""
    name = raw_name.replace("tBodyAcc", "Accelerometer")
    name = name.replace("tGravityAcc", "Gravity Accelerometer")
    name = name.replace("tBodyGyro", "Gyroscope")
    name = name.replace("tBodyAccJerk", "Accelerometer Jerk")
    name = name.replace("tBodyGyroJerk", "Gyroscope Jerk")
    name = name.replace("fBodyAcc", "Frequency Accelerometer")
    name = name.replace("fBodyGyro", "Frequency Gyroscope")
    name = name.replace("Mag", " Magnitude")
    name = name.replace("mean()", "Mean")
    name = name.replace("std()", "Std")
    name = name.replace("meanFreq()", "Mean Frequency")
    name = name.replace("-", " ")
    return name.strip()

def canonical(s: str) -> str:
    """Lowercase and remove non-alphanumeric."""
    return re.sub(r'[^0-9a-z]+', '', str(s).lower())

def load_image(path):
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        with open(abs_path, "rb") as img_file:
            img_bytes = img_file.read()
        encoded = base64.b64encode(img_bytes).decode()
        ext = abs_path.split(".")[-1].lower()
        mime = "png" if ext == "png" else "jpeg"
        st.markdown(f'<img src="data:image/{mime};base64,{encoded}" style="width:100%;">', unsafe_allow_html=True)
    else:
        st.info("Banner image not found (optional).")

# -----------------------------
# Load models & supporting files
# -----------------------------
model_svm = safe_joblib_load(MODEL_SVM_PATH)
model_rf = safe_joblib_load(MODEL_RF_PATH)
model_knn = safe_joblib_load(MODEL_KNN_PATH)
label_encoder = safe_joblib_load(ENCODER_PATH)

# load selected features (expected 0-based indices)
selected_features = np.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else np.arange(561)
selected_features = np.array(selected_features, dtype=int)
if selected_features.min() >= 1 and selected_features.max() == 561:
    selected_features -= 1  # convert 1-based to 0-based

train_mean = np.load(MEAN_PATH) if os.path.exists(MEAN_PATH) else np.zeros(561)
train_mean = np.array(train_mean, dtype=float)
chart_df = pd.read_csv(CHART_PATH) if os.path.exists(CHART_PATH) else None

# Feature names
raw_feature_names, feature_names = {}, {}
if os.path.exists(FEATURES_TXT_PATH):
    with open(FEATURES_TXT_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0]) - 1
                raw_feature_names[idx] = parts[1]
                feature_names[idx] = readable_name(parts[1])
if not raw_feature_names:
    raw_feature_names = {i: f"f{i+1}" for i in range(561)}
if not feature_names:
    feature_names = {i: f"Feature {i}" for i in range(561)}

# -----------------------------
# Prediction Helper
# -----------------------------
def predict_activity(X_input, model_choice):
    """Run prediction for the selected model(s)."""
    models = {
        "SVM": model_svm,
        "Random Forest": model_rf,
        "KNN": model_knn
    }

    results = {}
    for name, model in models.items():
        if model is None or (model_choice != name and model_choice != "Compare All"):
            continue
        try:
            pred = model.predict(X_input)
            prob = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None
            activity = label_encoder.inverse_transform(pred)[0] if label_encoder else pred[0]
            top3_idx = np.argsort(prob)[::-1][:3] if prob is not None else []
            top3_labels = label_encoder.inverse_transform(top3_idx) if (prob is not None and label_encoder) else []
            top3_probs = prob[top3_idx] if prob is not None else []

            results[name] = {
                "activity": activity,
                "top3_labels": list(top3_labels),
                "top3_probs": list(top3_probs)
            }
        except Exception as e:
            st.error(f"{name} prediction error: {e}")
    return results

def display_results(results):
    """Show prediction results and charts."""
    if not results:
        st.error("⚠ No model available or prediction failed.")
    for model_name, res in results.items():
        st.subheader(f"✅ {model_name} Predicted Activity: {res.get('activity', '')}")
        if res.get("top3_labels") and res.get("top3_probs"):
            df = pd.DataFrame({"Activity": res["top3_labels"], "Probability": res["top3_probs"]})
            st.bar_chart(df.set_index("Activity"))
    if chart_df is not None:
        st.subheader("📊 Activity Chart Values")
        st.dataframe(chart_df)

# -----------------------------
# UI Layout
# -----------------------------
tabs = st.tabs(["🏠 Home", "📂 Predict", "ℹ About Us", "📞 Contact Us"])

with tabs[0]:
    st.title("🤖 Human Activity Recognition (HAR)")
    st.markdown("""
    Welcome to the Human Activity Recognition App!

    Features:
    - Predict activities using SVM, Random Forest, or KNN
    - Manual input or File upload (CSV, Excel, TXT)
    - Compare Top-3 prediction probabilities from all models
    - Interactive activity charts for reference
    """)
    load_image(BANNER_IMAGE_PATH)

with tabs[1]:
    st.title("📂 Predict Human Activities")
    model_options = ["SVM", "Random Forest", "KNN", "Compare All"]
    model_choice = st.radio("Choose Model:", model_options, horizontal=True)
    mode = st.radio("Input Mode:", ["🖐 Manual Input", "📂 Upload File"], horizontal=True)

    user_features = np.array(train_mean, dtype=float)

    # Manual input
    if mode == "🖐 Manual Input":
        st.subheader("📊 Input Features (first 40 shown)")
        for idx in selected_features[:40]:
            fname = feature_names.get(idx, f"Feature {idx}")
            user_features[idx] = st.slider(fname, -10.0, 10.0, float(train_mean[idx]), 0.01)

    # File upload
    else:
        st.subheader("📂 Upload Files for Batch Prediction")
        uploaded_X = st.file_uploader("Upload X_test.txt", type=["txt"], key="X_test")
        uploaded_y = st.file_uploader("Upload y_test.txt", type=["txt"], key="y_test")

        if uploaded_X and uploaded_y:
            try:
                X_df = pd.read_csv(uploaded_X, delim_whitespace=True, header=None)
                y_df = pd.read_csv(uploaded_y, delim_whitespace=True, header=None)
                st.success(f"✅ Files loaded: X_test ({X_df.shape}), y_test ({y_df.shape})")
                row_idx = st.number_input("Row index (0-based)", 0, X_df.shape[0]-1, 0)
                user_features[selected_features] = X_df.iloc[row_idx, selected_features]
            except Exception as e:
                st.error(f"❌ Error reading files: {e}")

    # Predict
    if st.button("🔮 Predict Activity"):
        try:
            X_input = user_features[selected_features].reshape(1, -1)
            results = predict_activity(X_input, model_choice)
            display_results(results)
        except Exception as e:
            st.error(f"Error creating input vector: {e}")

# -----------------------------
# About & Contact
# -----------------------------
with tabs[2]:
    st.title("ℹ About Us")
    st.markdown("""
    Human Activity Recognition Project Team

    - Predicts human activities using accelerometer and gyroscope sensor data  
    - Developed with Python, Streamlit, and Scikit-learn  
    - Supports manual input and File predictions  
    - Compare predictions across SVM, Random Forest, and KNN  
    """)

with tabs[3]:
    st.title("📞 Contact Us")
    st.markdown("""
    Team Members:
    1. Dhange Sakshi  
    2. Raut Chaitali  
    3. Handge Sakshi  
    4. Patil Vaishnavi  
    5. More Dipali  

    Reach Us:
    - Email: chaitaliraut2005@gmail.com 
    - Phone: +91 9021812907  
    - GitHub: [Repo](https://github.com/chaitali-n-raut/G14_Human_Motion_Recognition)  
    - LinkedIn: [LinkedIn](https://www.linkedin.com/in/chaitali-raut-3403cr/)  
    """)
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message")
        submit = st.form_submit_button("Send Message")
        if submit:
            st.success("✅ Thank you! Your message has been sent.")
