# =============================================================================
# ADULT INCOME PREDICTION - COMPLETE ML APPLICATION
# =============================================================================
# Dataset: UCI Adult / Census Income Dataset
# Target:  income (binary: <=50K or >50K)
# Author:  Senior ML Engineer
# =============================================================================

import os
import io
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score,
                             roc_curve, f1_score, precision_score,
                             recall_score)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
# DATA_PATH  = "/mnt/user-data/uploads/adult_3.csv"
DATA_PATH = r"C:\Users\HP\OneDrive\Documents\Desktop\Salary-Prediction-Model\files\adult 3.csv"
MODEL_PATH = "best_model.pkl"
META_PATH  = "model_meta.pkl"

PALETTE = px.colors.qualitative.Set2

# =============================================================================
# 1. DATA LOADING & INSPECTION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_raw_data(path: str) -> pd.DataFrame:
    """Load the CSV file; raise a clear error if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded dataset: %s rows × %s cols", *df.shape)
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a structured summary of the raw dataset."""
    # Detect '?' as missing
    df_tmp = df.copy()
    for col in df_tmp.select_dtypes(include=["object"]).columns:  #, "str"
        df_tmp[col] = df_tmp[col].str.strip().replace("?", np.nan)

    numerical   = df_tmp.select_dtypes(include=["number"]).columns.tolist()
    categorical = df_tmp.select_dtypes(exclude=["number"]).columns.tolist()

    # Identify target: last column by convention OR first binary-ish col
    target = df_tmp.columns[-1]

    return {
        "shape"       : df.shape,
        "columns"     : df.columns.tolist(),
        "dtypes"      : df.dtypes.astype(str).to_dict(),
        "numerical"   : numerical,
        "categorical" : categorical,
        "target"      : target,
        "missing"     : df_tmp.isnull().sum().to_dict(),
        "missing_pct" : (df_tmp.isnull().sum() / len(df_tmp) * 100).round(2).to_dict(),
        "duplicates"  : int(df.duplicated().sum()),
        "stats"       : df_tmp[numerical].describe().round(3),
        "df_clean_preview": df_tmp.head(5),
    }

# =============================================================================
# 2. DATA CLEANING & PREPROCESSING
# =============================================================================

@st.cache_data(show_spinner=False)
def clean_and_encode(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    1. Strip whitespace
    2. Replace '?' with NaN → fill with mode
    3. Drop duplicates
    4. Cap outliers (IQR × 3)
    5. Label-encode categoricals
    6. StandardScale numericals (excl. target)
    Returns (X_train, X_test, y_train, y_test, feature_names, label_map,
             encoders, scaler, processed_df)
    """
    df = df.copy()

    # --- 1. Strip whitespace from string columns ---
    for col in df.select_dtypes(include=["object"]).columns: #, "str"
        df[col] = df[col].str.strip()

    # --- 2. Replace '?' with NaN, fill with mode ---
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # --- 3. Drop duplicates ---
    df.drop_duplicates(inplace=True)

    # --- 4. IQR outlier capping (only numerical, not target) ---
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 3 * iqr, upper=q3 + 3 * iqr)

    # --- 5. Identify target ---
    target = df.columns[-1]

    # --- 6. Encode target ---
    target_le  = LabelEncoder()
    y = target_le.fit_transform(df[target])
    label_map  = dict(zip(target_le.classes_,
                          target_le.transform(target_le.classes_)))

    # --- 7. Encode categorical features ---
    cat_cols = df.select_dtypes(exclude=["number"]).columns.drop(target).tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # --- 8. Scale numerical features ---
    feature_cols = [c for c in df.columns if c != target]
    num_feat     = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    scaler       = StandardScaler()
    df_scaled    = df[feature_cols].copy()
    df_scaled[num_feat] = scaler.fit_transform(df_scaled[num_feat])

    X = df_scaled.values
    feature_names = feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    return (X_train, X_test, y_train, y_test,
            feature_names, label_map, encoders, scaler, df)

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

def get_models() -> dict:
    """Return dictionary of candidate classifiers."""
    return {
        "Logistic Regression"     : LogisticRegression(max_iter=1000,
                                                        random_state=42),
        "Ridge Classifier"        : RidgeClassifier(),
        "Decision Tree"           : DecisionTreeClassifier(max_depth=8,
                                                           random_state=42),
        "Random Forest"           : RandomForestClassifier(n_estimators=100,
                                                           random_state=42,
                                                           n_jobs=-1),
        "Gradient Boosting"       : GradientBoostingClassifier(n_estimators=100,
                                                                random_state=42),
        "XGBoost"                 : XGBClassifier(n_estimators=100,
                                                  random_state=42,
                                                  eval_metric="logloss",
                                                  verbosity=0),
        "Linear SVC"              : LinearSVC(max_iter=2000, random_state=42),
    }


def train_all_models(X_train, X_test, y_train, y_test,
                     feature_names) -> pd.DataFrame:
    """Train every model, evaluate, return results DataFrame + trained models."""
    models   = get_models()
    results  = []
    trained  = {}

    progress = st.progress(0, text="Training models…")
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}…"):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Some models (SVC, Ridge) don't have predict_proba
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    roc    = round(roc_auc_score(y_test, y_prob), 4)
                else:
                    roc = "N/A"

                acc  = round(accuracy_score(y_test, y_pred),  4)
                f1   = round(f1_score(y_test, y_pred),        4)
                prec = round(precision_score(y_test, y_pred), 4)
                rec  = round(recall_score(y_test, y_pred),    4)

                results.append({
                    "Model"     : name,
                    "Accuracy"  : acc,
                    "F1 Score"  : f1,
                    "Precision" : prec,
                    "Recall"    : rec,
                    "ROC-AUC"   : roc,
                })
                trained[name] = model
                logger.info("%s → Acc: %.4f  F1: %.4f", name, acc, f1)
            except Exception as e:
                logger.error("Failed to train %s: %s", name, e)
                st.warning(f"{name} failed: {e}")

        progress.progress((i + 1) / len(models),
                          text=f"Done: {name}")

    progress.empty()
    results_df = pd.DataFrame(results).sort_values("Accuracy",
                                                    ascending=False).reset_index(drop=True)
    return results_df, trained

# =============================================================================
# 4. BEST MODEL SELECTION & PERSISTENCE
# =============================================================================

def save_best_model(trained: dict, results_df: pd.DataFrame,
                    feature_names: list, encoders: dict,
                    scaler, label_map: dict):
    """Pickle the best model and its metadata."""
    best_name  = results_df.iloc[0]["Model"]
    best_model = trained[best_name]

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    meta = {
        "best_name"    : best_name,
        "feature_names": feature_names,
        "encoders"     : encoders,
        "scaler"       : scaler,
        "label_map"    : label_map,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    logger.info("Saved best model: %s", best_name)
    return best_name


def load_saved_model():
    """Load pickled model + metadata if available."""
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(META_PATH, "rb") as f:
            meta  = pickle.load(f)
        logger.info("Loaded saved model: %s", meta["best_name"])
        return model, meta
    return None, None

# =============================================================================
# 5. FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from tree-based or coef-based models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": imp
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df_imp

# =============================================================================
# 6. STREAMLIT PAGE RENDERERS
# =============================================================================

# ─── PAGE: HOME ─────────────────────────────────────────────────────────────

def page_home():
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <h1 style="font-size:3rem; font-weight:800;">💰 Income Predictor</h1>
        <p style="font-size:1.2rem; color:#888;">
            End-to-end Machine Learning on the UCI Adult Census Dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📋 Records",   "48 842",  "after cleaning")
    with c2:
        st.metric("🔢 Features",  "14",       "input columns")
    with c3:
        st.metric("🤖 Models",    "7",        "algorithms")
    with c4:
        st.metric("🎯 Task",      "Binary",   "Classification")

    st.divider()

    st.markdown("### 🗺️ Project Workflow")
    cols = st.columns(4)
    steps = [
        ("🔍", "Data Understanding",
         "Load, inspect, and summarise the dataset automatically."),
        ("🧹", "Data Cleaning",
         "Handle '?'-encoded missing values, duplicates, and outliers."),
        ("📊", "EDA Dashboard",
         "Interactive Plotly charts — distributions, correlations, pair plots."),
        ("🤖", "ML Pipeline",
         "Train 7 classifiers, compare metrics, auto-select the best model."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style='padding:1rem; border-radius:12px;
                    border:1px solid #333; text-align:center;'>
            <div style='font-size:2rem'>{icon}</div>
            <strong>{title}</strong>
            <p style='font-size:.85rem; color:#888; margin-top:.5rem'>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.info("👈  Use the sidebar to navigate between sections.")


# ─── PAGE: DATASET OVERVIEW ─────────────────────────────────────────────────

def page_dataset_overview(df: pd.DataFrame, info: dict):
    st.title("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",       info["shape"][0])
    c2.metric("Columns",    info["shape"][1])
    c3.metric("Duplicates", info["duplicates"])
    c4.metric("Target",     info["target"])

    st.subheader("🔎 Raw Sample (first 500 rows)")
    st.dataframe(df.head(500), use_container_width=True)

    st.subheader("📐 Data Types")
    dtype_df = pd.DataFrame({
        "Column"  : list(info["dtypes"].keys()),
        "Type"    : list(info["dtypes"].values()),
        "Category": ["Numerical" if c in info["numerical"]
                     else "Categorical"
                     for c in info["dtypes"].keys()],
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("📈 Statistical Summary")
    st.dataframe(info["stats"], use_container_width=True)

    st.subheader("❓ Missing Values (encoded as '?')")
    miss = pd.DataFrame({
        "Column" : list(info["missing"].keys()),
        "Missing": list(info["missing"].values()),
        "Pct %"  : list(info["missing_pct"].values()),
    }).query("Missing > 0")

    if miss.empty:
        st.success("No null values — but '?' codes exist in 3 columns "
                   "(workclass, occupation, native-country). "
                   "They are replaced by the column mode during cleaning.")
    else:
        fig = px.bar(miss, x="Column", y="Pct %",
                     title="Missing Value %", color="Column",
                     color_discrete_sequence=PALETTE)
        st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: DATA ANALYSIS ─────────────────────────────────────────────────────

def page_data_analysis(df: pd.DataFrame, info: dict):
    st.title("🔬 Data Analysis")

    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns: #, "str"
        df_clean[col] = df_clean[col].str.strip().replace("?", np.nan)
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    target = info["target"]
    num_cols = info["numerical"]
    cat_cols = [c for c in info["categorical"] if c != target]

    # Target distribution
    st.subheader("🎯 Target Distribution")
    vc = df_clean[target].value_counts().reset_index()
    vc.columns = [target, "count"]
    fig = px.pie(vc, names=target, values="count",
                 color_discrete_sequence=PALETTE,
                 title=f"Distribution of '{target}'")
    st.plotly_chart(fig, use_container_width=True)

    # Numerical distributions
    st.subheader("📊 Numerical Feature Distributions")
    chosen_num = st.selectbox("Select numerical feature:", num_cols, key="num_dist")
    fig = px.histogram(df_clean, x=chosen_num, color=target,
                       barmode="overlay", nbins=40,
                       color_discrete_sequence=PALETTE,
                       title=f"{chosen_num} by {target}")
    st.plotly_chart(fig, use_container_width=True)

    # Boxplots
    st.subheader("📦 Boxplots (Numerical vs Target)")
    chosen_box = st.selectbox("Select feature for boxplot:", num_cols, key="box")
    fig = px.box(df_clean, x=target, y=chosen_box, color=target,
                 color_discrete_sequence=PALETTE,
                 title=f"{chosen_box} grouped by {target}")
    st.plotly_chart(fig, use_container_width=True)

    # Categorical analysis
    st.subheader("🏷️ Categorical Feature Analysis")
    chosen_cat = st.selectbox("Select categorical feature:", cat_cols, key="cat")
    ct = df_clean.groupby([chosen_cat, target]).size().reset_index(name="count")
    fig = px.bar(ct, x=chosen_cat, y="count", color=target, barmode="group",
                 color_discrete_sequence=PALETTE,
                 title=f"{chosen_cat} vs {target}")
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Age vs Hours scatter
    st.subheader("🔵 Age vs Hours-per-Week")
    sample = df_clean.sample(min(3000, len(df_clean)), random_state=42)
    if "age" in sample.columns and "hours-per-week" in sample.columns:
        fig = px.scatter(sample, x="age", y="hours-per-week",
                         color=target, opacity=0.5,
                         color_discrete_sequence=PALETTE,
                         title="Age vs Hours-per-Week (sampled)")
        st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: VISUALIZATIONS ────────────────────────────────────────────────────

def page_visualizations(df: pd.DataFrame, info: dict):
    st.title("📊 Visualizations")

    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns: #, "str"
        df_clean[col] = df_clean[col].str.strip().replace("?", np.nan)
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    num_cols = info["numerical"]
    target   = info["target"]

    # Correlation heatmap
    st.subheader("🌡️ Correlation Heatmap")
    corr = df_clean[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Pearson Correlation — Numerical Features")
    st.plotly_chart(fig, use_container_width=True)

    # Distribution grid
    st.subheader("📉 All Numerical Distributions")
    n = len(num_cols)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig = make_subplots(rows=nrows_grid, cols=ncols_grid,
                        subplot_titles=num_cols)
    for idx, col in enumerate(num_cols):
        r, c = divmod(idx, ncols_grid)
        vals = df_clean[col].dropna()
        fig.add_trace(go.Histogram(x=vals, name=col,
                                   marker_color=PALETTE[idx % len(PALETTE)],
                                   showlegend=False),
                      row=r + 1, col=c + 1)
    fig.update_layout(height=300 * nrows_grid, title_text="Numerical Distributions")
    st.plotly_chart(fig, use_container_width=True)

    # Pair plot (sampled)
    st.subheader("🔗 Pair Plot (sampled, up to 4 features)")
    pair_cols = num_cols[:4]
    sample    = df_clean.sample(min(2000, len(df_clean)), random_state=0)
    fig = px.scatter_matrix(sample, dimensions=pair_cols, color=target,
                             color_discrete_sequence=PALETTE,
                             title="Scatter Matrix")
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig, use_container_width=True)

    # Gender / race distribution
    for cat in ["gender", "race"]:
        if cat in df_clean.columns:
            st.subheader(f"👥 {cat.title()} Distribution by Income")
            ct = df_clean.groupby([cat, target]).size().reset_index(name="count")
            fig = px.bar(ct, x=cat, y="count", color=target, barmode="group",
                         color_discrete_sequence=PALETTE,
                         title=f"{cat.title()} vs {target}")
            st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: MODEL TRAINING ────────────────────────────────────────────────────

def page_model_training(df: pd.DataFrame, info: dict):
    st.title("🤖 Model Training")

    st.info(
        "Click **Train Models** to run all 7 classifiers. "
        "Results are cached — re-training only happens if you click again."
    )

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Preprocessing data…"):
            (X_train, X_test, y_train, y_test,
             feat_names, label_map, encoders, scaler, _) = clean_and_encode(df)

        results_df, trained = train_all_models(
            X_train, X_test, y_train, y_test, feat_names
        )

        best_name = save_best_model(
            trained, results_df, feat_names, encoders, scaler, label_map
        )

        st.session_state["results_df"] = results_df
        st.session_state["trained"]    = trained
        st.session_state["feat_names"] = feat_names
        st.session_state["X_test"]     = X_test
        st.session_state["y_test"]     = y_test
        st.session_state["best_name"]  = best_name
        st.success(f"✅ Training complete! Best model: **{best_name}**")

    # Show results if available
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]

        st.subheader("📊 Model Comparison")
        st.dataframe(
            results_df.style.highlight_max(
                subset=["Accuracy", "F1 Score", "Precision", "Recall"],
                color="#1a6b2a",
            ),
            use_container_width=True,
        )

        # Bar chart comparison
        fig = px.bar(
            results_df.melt(id_vars="Model",
                            value_vars=["Accuracy", "F1 Score",
                                        "Precision", "Recall"]),
            x="Model", y="value", color="variable", barmode="group",
            color_discrete_sequence=PALETTE,
            title="Model Metrics Comparison",
        )
        fig.update_xaxes(tickangle=-20)
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        # Best model highlight
        st.success(f"🏆 Best Model: **{st.session_state['best_name']}**  "
                   f"(Accuracy = {results_df.iloc[0]['Accuracy']})")

        # Feature importance
        best_model = st.session_state["trained"][st.session_state["best_name"]]
        imp_df     = get_feature_importance(best_model,
                                            st.session_state["feat_names"])
        if not imp_df.empty:
            st.subheader("🔑 Feature Importance (Best Model)")
            fig = px.bar(imp_df.head(14), x="Importance", y="Feature",
                         orientation="h", color="Importance",
                         color_continuous_scale="Teal",
                         title="Top Feature Importances")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        # Download results
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️  Download Results CSV",
            csv_buf.getvalue(),
            file_name="model_results.csv",
            mime="text/csv",
        )


# ─── PAGE: SALARY PREDICTION ─────────────────────────────────────────────────

def page_prediction(df: pd.DataFrame, info: dict):
    st.title("🎯 Income Prediction")

    # Try to load saved model
    model, meta = load_saved_model()

    if model is None:
        st.warning("⚠️  No trained model found. "
                   "Please go to **Model Training** and click Train.")
        return

    st.success(f"✅ Loaded: **{meta['best_name']}**")

    encoders     = meta["encoders"]
    scaler       = meta["scaler"]
    feature_names = meta["feature_names"]
    label_map    = meta["label_map"]
    inv_label    = {v: k for k, v in label_map.items()}

    # --- Build input widgets dynamically ---
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns: #, "str"
        df_clean[col] = df_clean[col].str.strip().replace("?", np.nan)
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    target = info["target"]
    st.subheader("✏️  Enter Feature Values")

    input_values = {}
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        col_widget = cols[i % 2]
        if feat in encoders:                          # categorical
            le      = encoders[feat]
            options = list(le.classes_)
            sel     = col_widget.selectbox(feat, options, key=f"inp_{feat}")
            input_values[feat] = le.transform([sel])[0]
        else:                                          # numerical
            mn  = float(df_clean[feat].min()) if feat in df_clean else 0.0
            mx  = float(df_clean[feat].max()) if feat in df_clean else 100.0
            med = float(df_clean[feat].median()) if feat in df_clean else (mn + mx) / 2
            val = col_widget.slider(feat, min_value=mn, max_value=mx,
                                    value=med, key=f"inp_{feat}")
            input_values[feat] = val

    if st.button("🔮 Predict Income", type="primary"):
        # Build feature array — apply same scaling
        raw_arr = np.array([[input_values[f] for f in feature_names]])

        # Re-scale numerical columns
        num_feat_indices = [i for i, f in enumerate(feature_names)
                            if f not in encoders]
        # We need to pass a full array to the scaler
        # Reconstruct a dummy row with ONLY the columns the scaler expects
        # The scaler was fit on all feature_names (numerical subset internally)
        # To avoid mis-scaling we use scaler.transform on the whole row
        raw_arr_scaled = raw_arr.copy().astype(float)
        # Scaler was fit on only the numerical features in order;
        # categorical features were already label-encoded (integers).
        # We replicate: scaler transforms all columns (it was fit on all features)
        raw_arr_scaled = scaler.transform(raw_arr_scaled)

        pred  = model.predict(raw_arr_scaled)[0]
        label = inv_label.get(pred, str(pred))

        if hasattr(model, "predict_proba"):
            prob  = model.predict_proba(raw_arr_scaled)[0]
            conf  = max(prob) * 100
        else:
            conf  = None

        # Display result card
        color = "#27ae60" if label == ">50K" else "#e74c3c"
        st.markdown(f"""
        <div style='margin-top:1.5rem; padding:1.5rem; border-radius:16px;
                    border:2px solid {color}; text-align:center;'>
            <h2 style='color:{color}; font-size:2.5rem;'>
                {'💰 Income > $50K' if label == '>50K' else '📉 Income ≤ $50K'}
            </h2>
            <p style='font-size:1.2rem; color:#aaa;'>Predicted Class: <strong>{label}</strong></p>
            {'<p style="font-size:1rem; color:#888;">Confidence: <strong>'+f"{conf:.1f}%</strong></p>" if conf else ""}
        </div>
        """, unsafe_allow_html=True)

        # Save prediction to CSV for download
        pred_row = {**input_values,
                    "Predicted_Income": label,
                    "Confidence_%"    : f"{conf:.1f}" if conf else "N/A"}
        pred_df  = pd.DataFrame([pred_row])
        csv_buf  = io.StringIO()
        pred_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️  Download This Prediction",
            csv_buf.getvalue(),
            file_name="prediction.csv",
            mime="text/csv",
        )


# ─── PAGE: MODEL PERFORMANCE ─────────────────────────────────────────────────

def page_model_performance(df: pd.DataFrame, info: dict):
    st.title("📈 Model Performance")

    model, meta = load_saved_model()
    if model is None:
        st.warning("No saved model. Please train first.")
        return

    # Re-run preprocessing to get test set
    (X_train, X_test, y_train, y_test,
     feat_names, label_map, encoders, scaler, _) = clean_and_encode(df)

    y_pred   = model.predict(X_test)
    inv_map  = {v: k for k, v in label_map.items()}
    classes  = [inv_map[i] for i in sorted(inv_map)]

    st.subheader(f"🏆 Best Model: {meta['best_name']}")

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.4f}")
    c2.metric("F1 Score",  f"{f1:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall",    f"{rec:.4f}")

    # Confusion matrix
    st.subheader("🧩 Confusion Matrix")
    cm  = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True,
                    x=classes, y=classes,
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale="Blues",
                    title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    if hasattr(model, "predict_proba"):
        st.subheader("📉 ROC Curve")
        y_prob         = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _    = roc_curve(y_test, y_prob)
        auc            = roc_auc_score(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"ROC (AUC={auc:.4f})",
                                 line=dict(color="#2ecc71", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Random",
                                 line=dict(color="#888", dash="dash")))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                          title="ROC Curve", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=classes,
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    # Feature importance
    imp_df = get_feature_importance(model, feat_names)
    if not imp_df.empty:
        st.subheader("🔑 Feature Importance")
        fig = px.bar(imp_df.head(14), x="Importance", y="Feature",
                     orientation="h", color="Importance",
                     color_continuous_scale="Viridis",
                     title="Feature Importances")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: ABOUT ─────────────────────────────────────────────────────────────

def page_about():
    st.title("ℹ️ About This Project")
    st.markdown("""
## 💰 Adult Income Predictor

### Dataset
The **UCI Adult / Census Income** dataset contains demographic and employment
data from the 1994 US Census Bureau database.  
The prediction task is to determine whether a person earns **>50 K/year**.

| Attribute        | Detail                          |
|------------------|---------------------------------|
| Source           | UCI Machine Learning Repository |
| Rows (raw)       | 48 842                          |
| Features         | 14 (6 numerical, 8 categorical) |
| Target           | `income` (binary)               |
| Missing encoded  | `?` in workclass, occupation, native-country |

### Features Used
| Feature | Type | Description |
|---|---|---|
| age | Numerical | Age in years |
| workclass | Categorical | Employment type |
| fnlwgt | Numerical | Census weight |
| education | Categorical | Highest level of education |
| educational-num | Numerical | Education years |
| marital-status | Categorical | Marital status |
| occupation | Categorical | Job category |
| relationship | Categorical | Family role |
| race | Categorical | Race |
| gender | Categorical | Gender |
| capital-gain | Numerical | Capital gain |
| capital-loss | Numerical | Capital loss |
| hours-per-week | Numerical | Work hours/week |
| native-country | Categorical | Country of origin |

### Pipeline
```
Raw CSV → Strip & Replace '?' → Fill Mode → Drop Duplicates
         → IQR Outlier Cap → Label Encode → Standard Scale
         → Train/Test Split (80/20, stratified)
         → Train 7 Classifiers → Select Best → Pickle
```

### Models
- Logistic Regression  
- Ridge Classifier  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Linear SVC  

### Tech Stack
`Python 3.11` | `Streamlit` | `scikit-learn` | `XGBoost` |
`Pandas` | `NumPy` | `Plotly` | `Matplotlib` | `Seaborn`
    """)


# =============================================================================
# 7. MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Income Predictor ML",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global CSS
    st.markdown("""
    <style>
        .stApp { font-family: 'Inter', sans-serif; }
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 0.6rem 1rem;
        }
        .stButton>button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    # --- Load data ---
    try:
        raw_df = load_raw_data(DATA_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    info = inspect_dataset(raw_df)

    # --- Sidebar navigation ---
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/"
        "Dollar_sign_in_circle_clemson_2.svg/240px-Dollar_sign_in_circle_clemson_2.svg.png",
        width=80,
    )
    st.sidebar.title("💰 Income Predictor")
    st.sidebar.markdown("---")

    pages = {
        "🏠 Home"               : page_home,
        "📋 Dataset Overview"   : page_dataset_overview,
        "🔬 Data Analysis"      : page_data_analysis,
        "📊 Visualizations"     : page_visualizations,
        "🤖 Model Training"     : page_model_training,
        "🎯 Income Prediction"  : page_prediction,
        "📈 Model Performance"  : page_model_performance,
        "ℹ️  About Project"     : page_about,
    }

    selection = st.sidebar.radio("Navigate", list(pages.keys()))
    st.sidebar.markdown("---")

    # Dataset quick-stats in sidebar
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.markdown(f"**Rows:** {raw_df.shape[0]:,}")
    st.sidebar.markdown(f"**Cols:** {raw_df.shape[1]}")
    st.sidebar.markdown(f"**Target:** `{info['target']}`")
    model, meta = load_saved_model()
    if meta:
        st.sidebar.success(f"✅ Saved: {meta['best_name']}")
    else:
        st.sidebar.warning("⚠️ No trained model yet")

    # --- Render page ---
    fn = pages[selection]
    if selection in ("🏠 Home", "ℹ️  About Project"):
        fn()
    else:
        fn(raw_df, info)


if __name__ == "__main__":
    main()
