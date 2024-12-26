import streamlit as st
import os

# Page Config
st.set_page_config(
    page_title="Data Science Project Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title Section
st.markdown("""
<div style="background-color:#4CAF50;padding:15px;border-radius:10px">
<h1 style="color:white;text-align:center;">Welcome to the Data Science Project Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.write("")

# Group Introduction
st.subheader("👨‍👩‍👦‍👦 **Group Members**")
team_members = [
    "📌 Sou Pichchomrong",
    "📌 Sroeun Bunnarith",
    "📌 Sorng Seyha",
    "📌 Thorn Davin",
    "📌 Ngo Seakyarith"
]
for member in team_members:
    st.write(f"- {member}")

st.divider()

# Tabs for Organization
tab1, tab2, tab3 = st.tabs(["Introduction", "EDA", "Model Summary"])

with tab1:
    st.header("📄 **Introduction**")
    st.markdown("""
This is the introduction tab, where you can explore the project overview and objectives.

This project aims to analyze the impact of various advertising budgets on sales performance using advanced machine learning models. By utilizing datasets with features such as TV, Radio, and Newspaper budgets, this dashboard provides predictive insights and helps identify the most influential factors driving sales.

Explore the sections to dive deeper into exploratory data analysis, model performances, and more.
""")

with tab2:
    # Visualizations Section
    st.header("📊 **Exploratory Data Analysis**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Heatmap")
        heatmap_path = "images/correlation_heatmap.png"
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Correlation Heatmap", use_container_width=True)
        else:
            st.warning("Correlation Heatmap image not found.")

        st.subheader("Feature Importance")
        importance_path = "images/feature_importance.png"
        if os.path.exists(importance_path):
            st.image(importance_path, caption="Feature Importance", use_container_width=True)
        else:
            st.warning("Feature Importance image not found.")

    with col2:
        st.subheader("Pairplot")
        pairplot_path = "images/pairplot.png"
        if os.path.exists(pairplot_path):
            st.image(pairplot_path, caption="Pairplot of Variables", use_container_width=True)
        else:
            st.warning("Pairplot image not found.")

with tab3:
    # Model Training Summary
    st.header("📋 **Status of Each Model**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔹 **Linear Models**")
        st.markdown("""
        - Linear Regression ✅ (Trained)
        - Ridge Regression ✅ (Trained)
        - Lasso Regression ✅ (Trained)
        - ElasticNet Regression ✅ (Trained)
        """)

        st.subheader("🔹 **Bayesian Methods**")
        st.markdown("""
        - Bayesian Ridge Regression ✅ (Trained)
        """)

        st.subheader("🔹 **Other Specialized Models**")
        st.markdown("""
        - K-Nearest Neighbors (KNN) ✅ (Trained)
        """)

    with col2:
        st.subheader("🔹 **Tree-Based Models**")
        st.markdown("""
        - Decision Tree ✅ (Trained)
        - Random Forest ✅ (Trained)
        - Gradient Boosting ✅ (Trained)
        - Extreme Gradient Boosting (XGBoost) ❌ (Not trained; requires xgboost library)
        - LightGBM ❌ (Not trained; requires lightgbm library)
        - CatBoost ❌ (Not trained; requires catboost library)
        """)

        st.subheader("🔹 **Ensemble Methods**")
        st.markdown("""
        - AdaBoost ✅ (Trained)
        - Stacking Regressor ❌ (Not trained; can be implemented if needed)
        """)

        st.subheader("🔹 **Neural Networks**")
        st.markdown("""
        - Multi-Layer Perceptron (MLP) ✅ (Trained)
        - TensorFlow/Keras models ❌ (Not implemented; requires TensorFlow library)
        """)

st.divider()

# Navigation Button
if st.button("🚀 Go to Sales Predictor"):
    st.switch_page("pages/Sales_Predictor.py")

# Footer Section
st.write("")
st.markdown("""
<div style="text-align:center;color:gray;font-size:12px;">
    Built with ❤️ by Group 9 | Powered by Streamlit
</div>
""", unsafe_allow_html=True)