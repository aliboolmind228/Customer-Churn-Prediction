import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ---------------------------
# Page config & theme defaults
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction Dashboard", page_icon="📉", layout="wide")
st.markdown("""
<style>
.small-note {color:#6b7280; font-size:0.9rem;}
div[data-testid="stMetricValue"] {font-size: 1.75rem;}
</style>
""", unsafe_allow_html=True)

# Plotly global style (palette + template)
px.defaults.template = "plotly_white"  # try: plotly_dark / seaborn / ggplot2
px.defaults.color_discrete_sequence = px.colors.qualitative.Set3

# Consistent colors for Yes/No churn & heatmaps
CHURN_COLOR_MAP = {"Yes": "#4E9F3D", "No": "#FF0000"}   
HEATMAP_SCALE = "Turbo"  # or "Viridis", "Blues", "Cividis", "Plasma"

# ---------------------------
# Paths (change if needed)
# ---------------------------
DATA_PATH = "D:\Projects - Data Science\Projects - Machine Learning\Customer Churn Prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "D:\Projects - Data Science\Projects - Machine Learning\Customer Churn Prediction\churn_model.pkl"

# ---------------------------
# Data / model loaders
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df_enc = pd.get_dummies(df, drop_first=True)
    return df, df_enc

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

# ---------------------------
# Safe load
# ---------------------------
try:
    df_raw = load_data(DATA_PATH)
    df, df_encoded = preprocess(df_raw)
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Couldn't load data or model. Check file paths.\n\n{e}")
    st.stop()

FEATURE_COLS = [c for c in df_encoded.columns if c != "Churn"]

# ---------------------------
# Sidebar – global controls
# ---------------------------
st.sidebar.header("🔧 Controls")

ten_min, ten_max = int(df['tenure'].min()), int(df['tenure'].max())
tenure_range = st.sidebar.slider("Tenure (months)", ten_min, ten_max, (ten_min, ten_max))

chg_min, chg_max = float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max())
charges_range = st.sidebar.slider("Monthly Charges", float(chg_min), float(chg_max), (float(chg_min), float(chg_max)))

contract_opts = ['All'] + sorted(df_raw['Contract'].unique().tolist())
chosen_contract = st.sidebar.selectbox("Contract", options=contract_opts)

internet_opts = ['All'] + sorted(df_raw['InternetService'].unique().tolist())
chosen_internet = st.sidebar.selectbox("Internet Service", options=internet_opts)

thresh = st.sidebar.slider(
    "Churn probability threshold",
    0.1, 0.9, 0.5, 0.05,
    help="Predicted probability ≥ threshold ⇒ Churn = 1"
)

# Apply filters to EDA dataframe
mask = (
    (df['tenure'].between(tenure_range[0], tenure_range[1])) &
    (df['MonthlyCharges'].between(charges_range[0], charges_range[1]))
)
if chosen_contract != 'All':
    mask &= (df_raw.loc[df.index, 'Contract'] == chosen_contract)
if chosen_internet != 'All':
    mask &= (df_raw.loc[df.index, 'InternetService'] == chosen_internet)

df_filt = df.loc[mask].copy()
df_filt_raw = df_raw.loc[df_filt.index].copy()  # same rows with original string cats

# ---------------------------
# Header + KPIs
# ---------------------------
st.title("📉 Customer Churn Prediction Dashboard")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Customers", f"{len(df):,}")
with k2: st.metric("Churned (filtered)", int(df_filt['Churn'].sum()))
with k3: st.metric("Churn Rate (filtered)", f"{(df_filt['Churn'].mean()*100):.1f}%")
with k4: st.metric("Avg. Monthly Charges", f"${df_filt['MonthlyCharges'].mean():.2f}")
st.markdown('<div class="small-note">Use the sidebar filters to explore segments.</div>', unsafe_allow_html=True)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore", "Predict", "Model Performance"])

# === Overview tab ===
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        churn_counts = df_filt['Churn'].value_counts().rename({0: "No", 1: "Yes"})
        fig = go.Figure(
            data=[go.Pie(
                labels=churn_counts.index,
                values=churn_counts.values,
                hole=.55,
                marker=dict(colors=["#76053E", "#FF7F00"])
            )]
        )
        fig.update_layout(title="Churn Distribution (Filtered)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tmp = df_filt_raw.assign(Churn=df_filt['Churn'].values)
        rate_by_contract = tmp.groupby('Contract')['Churn'].mean().reset_index()
        fig2 = px.bar(
            rate_by_contract, x='Contract', y='Churn',
            title="Churn Rate by Contract",
            color='Contract',
            color_discrete_map={
                    "Month-to-month": "#E19898",  # amber
                    "One year": "#F39F5A",        # blue
                    "Two year": "#C147E9"         # green
    }
            
        )
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_layout(bargap=0)  # 5% gap between bars
        st.plotly_chart(fig2, use_container_width=True)

# === Explore tab ===
with tab2:
    st.subheader("Interactive Exploration")

    cat_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    cat_cols = [c for c in cat_cols if c in df_filt_raw.columns]

    left, right = st.columns([1,2])
    with left:
        chosen_cat = st.selectbox("Choose a category", options=cat_cols)
    with right:
        tmp = df_filt_raw.assign(Churn=df_filt['Churn'].values)
        rate = tmp.groupby(chosen_cat)['Churn'].mean().reset_index()
        fig3 = px.bar(
            rate, x=chosen_cat, y='Churn',
            title=f"Churn Rate by {chosen_cat}",
            color=chosen_cat,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Charges vs Tenure (colored by Churn)")
    fig4 = px.scatter(
        df_filt, x='tenure', y='MonthlyCharges',
        color=df_filt['Churn'].map({0: 'No', 1: 'Yes'}),
        labels={'color': 'Churn'},
        opacity=0.75
    )
    # Enforce our churn colors
    for tr in fig4.data:
        tr.marker.color = CHURN_COLOR_MAP.get(tr.name, tr.marker.color)
    st.plotly_chart(fig4, use_container_width=True)

    st.download_button(
        label="⬇️ Download filtered rows (CSV)",
        data=df_filt_raw.assign(Churn=df_filt['Churn']).to_csv(index=False).encode('utf-8'),
        file_name="filtered_customers.csv",
        mime="text/csv"
    )

# === Predict tab ===
with tab3:
    st.subheader("Predict Churn for a New Customer")

    form = st.form("predict_form")
    c1, c2, c3 = form.columns(3)
    with c1:
        tenure_in = form.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_in = form.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
    with c2:
        contract_in = form.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_in = form.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c3:
        techsupport_in = form.selectbox("Tech Support", ["Yes", "No"])
        paperless_in = form.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = form.form_submit_button("🔍 Predict")

    # Prepare zero vector matching training columns (excluding Churn)
    X_cols = FEATURE_COLS
    input_vec = pd.DataFrame([np.zeros(len(X_cols))], columns=X_cols)

    # Fill numeric features
    if 'tenure' in X_cols: input_vec.at[0, 'tenure'] = tenure_in
    if 'MonthlyCharges' in X_cols: input_vec.at[0, 'MonthlyCharges'] = monthly_in

    # Helper for one-hot (drop_first=True behavior)
    def set_onehot(prefix, value, options):
        base = options[0]  # dropped category
        for opt in options[1:]:
            col = f"{prefix}_{opt}"
            if col in input_vec.columns:
                input_vec.at[0, col] = 1 if value == opt else 0

    set_onehot("Contract", contract_in, ["Month-to-month", "One year", "Two year"])
    set_onehot("InternetService", internet_in, ["DSL", "Fiber optic", "No"])
    set_onehot("TechSupport", techsupport_in, ["No", "Yes"])
    if 'PaperlessBilling_Yes' in input_vec.columns:
        input_vec.at[0, 'PaperlessBilling_Yes'] = 1 if paperless_in == "Yes" else 0

    if submitted:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_vec)[0][1])
        else:
            # fallback if model lacks proba
            prob = float(model.predict(input_vec)[0])

        pred = 1 if prob >= thresh else 0

        st.markdown("#### Predicted Churn Probability")
        st.progress(min(max(prob, 0.0), 1.0))
        st.write(f"**Probability:** `{prob:.3f}`  |  **Threshold:** `{thresh:.2f}`")

        if pred == 1:
            st.error("⚠️ This customer is **likely to churn**.")
        else:
            st.success("✅ This customer is **likely to stay**.")

        st.caption("Tip: adjust the threshold in the sidebar to tune recall vs precision.")

# === Model Performance tab ===
with tab4:
    st.subheader("Model Performance (hold-out split)")

    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba, auc = None, None

    y_pred_default = model.predict(X_test)
    y_pred_thresh = (y_proba >= thresh).astype(int) if y_proba is not None else y_pred_default

    L, R = st.columns(2)
    with L:
        st.markdown("**Classification Report (default threshold)**")
        st.code(classification_report(y_test, y_pred_default, digits=3))
        if auc is not None:
            st.metric("ROC-AUC", f"{auc:.3f}")

    with R:
        st.markdown("**Confusion Matrix (with sidebar threshold)**")
        cm = confusion_matrix(y_test, y_pred_thresh)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=HEATMAP_SCALE,
                           labels=dict(x="Predicted", y="Actual", color="Count"))
        fig_cm.update_layout(margin=dict(l=40,r=40,b=40,t=40))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Feature Importances")
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
        fig_imp = px.bar(
            importances[::-1],  # reverse for horizontal ascending
            orientation='h',
            title="Most Important Features",
            labels={'value': 'Importance', 'index': 'Feature'},
            color=importances[::-1].index,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importances not available for this model.")
