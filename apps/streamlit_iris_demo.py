"""Optional local demo: streamlit run apps/streamlit_iris_demo.py"""
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ML Model Explorer", page_icon="🧪", layout="wide")
st.title("🧪 Scikit-learn Model Explorer")
st.caption("A small educational demo — not a medical or production decision tool.")
data = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, stratify=data.target, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(X_train, y_train)
a, b = st.columns(2)
a.metric("Held-out accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.1%}")
b.metric("Test samples", len(X_test))
st.subheader("Try a sample input")
cols = st.columns(4)
values = []
for col, feature in zip(cols, data.feature_names, strict=False):
    values.append(col.number_input(feature, min_value=0.0, max_value=10.0, value=float(data.data[feature].median()), step=0.1))
if st.button("Predict species", type="primary"):
    sample = pd.DataFrame([values], columns=data.feature_names)
    pred = int(model.predict(sample)[0])
    probs = model.predict_proba(sample)[0]
    st.success(f"Predicted class: {data.target_names[pred]}")
    st.bar_chart(pd.Series(probs, index=data.target_names, name="Probability"))
st.info("Install Streamlit separately with: python -m pip install streamlit")
