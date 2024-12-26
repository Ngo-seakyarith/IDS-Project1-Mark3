import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # Added import
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Data
@st.cache_data
def load_data():
    data_path = "https://raw.githubusercontent.com/Ngo-seakyarith/IDS-Project1-Mark3/main/IDS_project_linear/data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    else:
        st.error("data.csv not found. Please ensure it's in the project directory.")
        return None

df = load_data()

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
    # Check if df is loaded
    if df is not None:
        st.header("📊 **Exploratory Data Analysis**")
        
        # Display Data Preview
        st.subheader("📋 Data Preview")
        st.write(df.head())
        
        st.markdown("---")  # Separator
        
        # Correlation Heatmap and Feature Importance in columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
            ax_corr.set_title("Correlation Heatmap of Features", fontsize=16)
            st.pyplot(fig_corr)
        
        with col2:
            st.subheader("📈 Feature Importance")
            
            # Ensure 'Sales' is your target variable
            target_variable = 'Sales'  # Change if different
            if target_variable in df.columns:
                X = df.drop(target_variable, axis=1)
                y = df[target_variable]
                
                # Check if all features are numeric
                if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
                    st.warning("All features should be numeric to compute feature importances.")
                else:
                    # Initialize and train the model
                    model = RandomForestRegressor(random_state=42)
                    model.fit(X, y)
                    
                    # Get feature importances
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)
                    
                    # Plot Feature Importances
                    fig_feat, ax_feat = plt.subplots(figsize=(8, 6))
                    sns.barplot(data=feature_importance_df, x="Importance", y="Feature", ax=ax_feat, palette="viridis")
                    ax_feat.set_title("Feature Importance Ranking", fontsize=16)
                    ax_feat.set_xlabel("Importance")
                    ax_feat.set_ylabel("Feature")
                    st.pyplot(fig_feat)
            else:
                st.warning(f"'{target_variable}' column not found in the dataset. Please verify the target variable name.")
        
        st.markdown("---")  # Separator
        
        # Pairplot
        st.subheader("🔗 Pairplot of Variables")
        
        # Select numeric columns for pairplot
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Not enough numeric columns to generate a pairplot.")
        else:
            # Optionally, allow users to select specific columns
            selected_columns = st.multiselect(
                "Select Columns for Pairplot (Optional)",
                options=numeric_columns,
                default=numeric_columns
            )
            
            if selected_columns:
                pairplot_df = df[selected_columns]
            else:
                pairplot_df = df[numeric_columns]
            
            # Generate Pairplot
            with st.spinner("Generating Pairplot..."):
                pairplot_fig = sns.pairplot(pairplot_df, diag_kind="kde", corner=True)
                st.pyplot(pairplot_fig)
    else:
        st.warning("Data not loaded. Please ensure `data.csv` is present and correctly formatted.")

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
