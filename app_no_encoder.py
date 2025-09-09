import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="💼 Salary Prediction (Advanced)", layout="centered")
st.title("💼 Salary Prediction App (Advanced)")

# --- Load model, features and scaler safely ---
try:
    import joblib
    model, feature_columns, scaler = joblib.load("model_no_encoder.pkl")
except Exception:
    import pickle
    with open("model_no_encoder.pkl", "rb") as f:
        model, feature_columns, scaler = pickle.load(f)

if model is None or feature_columns is None or scaler is None:
    st.error("❌ Could not load model, feature columns, and scaler from 'model_no_encoder.pkl'.")
    st.stop()

# --- User Input ---
def user_input_features():
    data = {
        'age': st.number_input("Age", 17, 90, 30),
        'workclass': st.selectbox("Workclass", [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc',
            'Federal-gov', 'Local-gov', 'State-gov',
            'Without-pay', 'Never-worked']),
        'fnlwgt': st.number_input("Fnlwgt", 10000, 1000000, 150000),
        'education': st.selectbox("Education", [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
            'Some-college', 'Assoc-acdm', 'Assoc-voc',
            'Doctorate', 'Prof-school', 'Preschool',
            '10th', '12th', '1st-4th', '5th-6th', '7th-8th']),
        'educational-num': st.slider("Education Number", 1, 16, 10),
        'marital-status': st.selectbox("Marital Status", [
            'Married-civ-spouse', 'Divorced', 'Never-married',
            'Separated', 'Widowed', 'Married-spouse-absent']),
        'occupation': st.selectbox("Occupation", [
            'Tech-support', 'Craft-repair', 'Other-service',
            'Sales', 'Exec-managerial', 'Prof-specialty',
            'Handlers-cleaners', 'Machine-op-inspct',
            'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv',
            'Protective-serv', 'Armed-Forces']),
        'relationship': st.selectbox("Relationship", [
            'Wife', 'Own-child', 'Husband', 'Not-in-family',
            'Other-relative', 'Unmarried']),
        'race': st.selectbox("Race", [
            'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
            'Other', 'Black']),
        'gender': st.selectbox("Gender", ['Female', 'Male']),
        'capital-gain': st.number_input("Capital Gain", min_value=0, value=0),
        'capital-loss': st.number_input("Capital Loss", min_value=0, value=0),
        'hours-per-week': st.slider("Hours per Week", 1, 100, 40),
        'native-country': st.selectbox("Native Country", [
            'United-States', 'India', 'Mexico', 'Philippines', 'Germany',
            'Canada', 'England', 'China', 'Cuba', 'Iran', 'South',
            'Vietnam', 'Puerto-Rico'])
    }
    return pd.DataFrame([data])

df_input = user_input_features()

# --- Predict & Visualize ---
if st.button("🔮 Predict Income"):
    try:
        # One-hot encode & align
        df_encoded = pd.get_dummies(df_input)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
        df_scaled = scaler.transform(df_encoded)

        # Predict
        prediction = model.predict(df_scaled)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df_scaled)[0][1]
        else:
            probability = None

        st.success(f"✅ Predicted Income: {prediction[0]}")
        if probability is not None:
            st.info(f"🧮 Probability (>50K): {probability:.2%}")
            st.metric("Confidence Level", f"{probability*100:.2f}%")

        # Feature importance (if logistic regression or linear model)
        if hasattr(model, "coef_"):
            st.subheader("📌 Top 10 Most Influential Features")
            feature_importance = pd.Series(np.abs(model.coef_[0]), index=feature_columns)
            top_features = feature_importance.sort_values(ascending=False).head(10)
            st.bar_chart(top_features)

        # SHAP Explainability
        st.subheader("💡 SHAP Local Explanation")
        df_for_shap = pd.DataFrame(df_scaled, columns=feature_columns)
        explainer = shap.Explainer(model, df_for_shap)
        shap_values = explainer(df_for_shap)

        st.subheader("🔍 SHAP Waterfall Plot (Single Prediction)")
        fig_local = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_local.figure)

        st.subheader("📊 SHAP Summary Plot (Global Feature Impact)")
        fig_summary = plt.figure()
        shap.summary_plot(shap_values, df_for_shap, show=False)
        st.pyplot(fig_summary)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

