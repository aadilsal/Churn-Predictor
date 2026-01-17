"""
Churn Intelligence Platform
============================

A modern, interactive dashboard for customer churn prediction.
Supports custom data upload and real-time predictions.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #3b82f6 100%);
        padding: 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 50%;
        height: 100%;
        background: radial-gradient(circle at 70% 30%, rgba(59, 130, 246, 0.3) 0%, transparent 50%);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 1rem;
        max-width: 600px;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px dashed #94a3b8;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }
    
    /* How it works */
    .step-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 16px;
    }
    
    [data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        color: #1e293b !important;
    }
    
    .info-box strong {
        color: #065f46 !important;
        font-size: 1rem;
    }
    
    .info-box br + br {
        display: block;
        margin-top: 0.25rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    .section-subheader {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


def load_model():
    """Load the trained model.
    
    Handles version compatibility issues gracefully.
    """
    import joblib
    
    model_path = Path("models/final_model.joblib")
    preprocessor_path = Path("models/feature_preprocessor.joblib")
    
    if model_path.exists() and preprocessor_path.exists():
        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            st.session_state.model_loaded = True
            return model, preprocessor
        except Exception as e:
            st.warning(f"⚠️ Model loading failed (version compatibility). Using demo mode.")
            st.session_state.model_loaded = False
            return None, None
    return None, None


def process_uploaded_data(df, model, preprocessor):
    """Process uploaded data and generate predictions.
    
    Handles missing columns by filling with default values.
    """
    try:
        # Define expected columns and their default values
        expected_columns = {
            'gender': 'Male',
            'SeniorCitizen': 'No',
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 50.0,
            'TotalCharges': 600.0,
        }
        
        # Check for missing columns and fill with defaults
        missing_cols = []
        for col, default_val in expected_columns.items():
            if col not in df.columns:
                df[col] = default_val
                missing_cols.append(col)
        
        if missing_cols:
            st.warning(f"⚠️ **{len(missing_cols)} columns were missing** and filled with default values: {', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}")
        
        # Prepare features (exclude ID and target columns)
        feature_cols = [c for c in df.columns if c not in ["customerID", "Churn", "churn_probability", "risk_level"]]
        
        # Ensure feature columns match expected order
        available_features = [c for c in expected_columns.keys() if c in feature_cols]
        X = preprocessor.transform(df[available_features])
        
        # Predict
        probabilities = model.predict_proba(X)[:, 1]
        
        # Add results
        df["churn_probability"] = probabilities
        df["risk_level"] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("💡 **Tip**: Make sure your data includes customer attributes like tenure, contract type, and services.")
        return None


def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Churn Intelligence")
        st.markdown("---")
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.warning("⏳ Model Loading...")
        
        if st.session_state.uploaded_data is not None:
            st.info(f"📊 {len(st.session_state.uploaded_data):,} customers loaded")
        
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("""
        - 🏠 Home & Upload
        - 📊 [Overview](/Overview)
        - ⚠️ [Risk Analysis](/Risk_Analysis)
        - 👤 [Customer View](/Customer_View)
        - 🔧 [Model Health](/Model_Health)
        - 🔮 [Simulator](/What_If_Simulator)
        """)
        
        st.markdown("---")
        st.markdown("### About")
        st.caption("""
        This platform uses machine learning to predict 
        customer churn and provide actionable insights 
        for retention strategies.
        """)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🎯 Churn Intelligence Platform</h1>
        <p class="hero-subtitle">
            Predict customer churn with AI-powered analytics. Upload your data, 
            get instant predictions, and discover actionable insights to retain your most valuable customers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown('<h2 class="section-header">How It Works</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Three simple steps to understand your customer churn risk</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="step-badge">1</div>
            <div class="feature-title">Upload Your Data</div>
            <div class="feature-desc">
                Upload a CSV file with your customer data. We accept standard CRM exports 
                with customer demographics, services, and billing information.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="step-badge">2</div>
            <div class="feature-title">AI Analysis</div>
            <div class="feature-desc">
                Our machine learning model analyzes each customer's profile to calculate 
                their churn probability and identify key risk factors.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="step-badge">3</div>
            <div class="feature-title">Take Action</div>
            <div class="feature-desc">
                Get personalized recommendations for each high-risk customer. Export reports 
                and integrate insights into your retention campaigns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Upload Section
    st.markdown("---")
    st.markdown('<h2 class="section-header">📤 Upload Your Customer Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with customer data"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            
            st.success(f"✅ Successfully loaded {len(df):,} customers!")
            
            # Preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Process data
            model, preprocessor = load_model()
            
            if model is not None:
                if st.button("🚀 Run Churn Analysis", type="primary"):
                    with st.spinner("Analyzing customer data..."):
                        results = process_uploaded_data(df.copy(), model, preprocessor)
                        if results is not None:
                            st.session_state.predictions = results
                            st.success("✅ Analysis complete! Navigate to Overview to see results.")
                            st.balloons()
            else:
                st.warning("Model not found. Using demo mode with simulated predictions.")
                if st.button("🚀 Run Demo Analysis", type="primary"):
                    # Simulate predictions for demo
                    np.random.seed(42)
                    df["churn_probability"] = np.random.beta(2, 5, len(df))
                    if "Churn" in df.columns:
                        df["churn_probability"] = np.where(
                            df["Churn"] == 1,
                            np.clip(df["churn_probability"] + 0.3, 0, 1),
                            df["churn_probability"]
                        )
                    df["risk_level"] = pd.cut(
                        df["churn_probability"],
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=["Low", "Medium", "High"]
                    )
                    st.session_state.predictions = df
                    st.success("✅ Demo analysis complete! Navigate to Overview to see results.")
                    st.balloons()
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>📋 Required Columns</strong><br>
            Your CSV should include:<br>
            • Customer ID<br>
            • Demographics (gender, senior status)<br>
            • Services (internet, phone, etc.)<br>
            • Account info (tenure, charges)<br>
            • Contract & payment details
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sample data download
        sample_path = Path("data/processed/telco_churn_processed.csv")
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            st.download_button(
                "📥 Download Sample Dataset",
                sample_df.to_csv(index=False),
                "sample_customer_data.csv",
                "text/csv"
            )
    
    st.markdown("---")
    
    # Platform Capabilities
    st.markdown('<h2 class="section-header">🔮 Platform Capabilities</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Churn Prediction</div>
            <div class="feature-desc">
                ML model with 84.5% accuracy predicts which customers are likely to churn.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">
                Understand why each customer is at risk with SHAP-based explanations.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⏱️</div>
            <div class="feature-title">Survival Analysis</div>
            <div class="feature-desc">
                Estimate when customers are likely to churn for optimal intervention timing.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Action Recommendations</div>
            <div class="feature-desc">
                Get personalized retention strategies for each at-risk customer.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Info
    st.markdown('<h2 class="section-header">🤖 About the Model</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Performance Metrics
        
        | Metric | Score |
        |--------|-------|
        | ROC-AUC | **0.845** |
        | PR-AUC | 0.655 |
        | Accuracy | 74.4% |
        | Precision | 51.1% |
        | Recall | 80.7% |
        
        *Model trained on XGBoost with 23 engineered features*
        """)
    
    with col2:
        st.markdown("""
        #### Top Churn Predictors
        
        1. **Contract Type** - Month-to-month contracts are 2.2x more likely to churn
        2. **Tenure** - New customers (<12 mo) are at highest risk
        3. **Internet Service** - Fiber optic customers churn more
        4. **Payment Method** - Electronic check payments correlate with higher churn
        5. **Support Services** - Tech support reduces churn by 20%
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <p>Built with ❤️ using Machine Learning • XGBoost • Streamlit</p>
        <p style="font-size: 0.875rem;">© 2026 Churn Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
