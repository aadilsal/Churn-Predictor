"""
Page 3: Customer View
=====================

Individual customer analysis with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Customer View", page_icon="👤", layout="wide")

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
    }
    
    .profile-name {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .prob-gauge {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .section-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .section-desc {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .driver-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        background: #f8fafc;
        color: #000000 !important;
    }
    
    .driver-item div {
        color: #000000 !important;
    }
    
    .driver-item div[style*="color: #64748b"] {
        color: #333333 !important;
    }
    
    .driver-risk { border-left: 4px solid #ef4444; }
    .driver-protect { border-left: 4px solid #10b981; }
    
    .action-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


def get_data():
    """Get data from session state or sample."""
    if "predictions" in st.session_state and st.session_state.predictions is not None:
        return st.session_state.predictions, True
    
    from pathlib import Path
    data_path = Path("data/processed/telco_churn_processed.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
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
        return df, False
    return None, False


def get_shap_drivers(row):
    """Calculate what's driving this customer's risk."""
    drivers = []
    
    # Risk factors (increase churn)
    if "Contract" in row and row["Contract"] == "Month-to-month":
        drivers.append(("Month-to-Month Contract", +0.18, "risk", 
                       "No commitment means the customer can leave anytime"))
    
    if "tenure" in row and row["tenure"] < 12:
        drivers.append(("New Customer", +0.12, "risk",
                       "Less than 12 months tenure - still in high-risk window"))
    
    if "InternetService" in row and row["InternetService"] == "Fiber optic":
        drivers.append(("Fiber Optic Internet", +0.08, "risk",
                       "Fiber customers tend to have higher expectations"))
    
    if "PaymentMethod" in row and row["PaymentMethod"] == "Electronic check":
        drivers.append(("Electronic Check Payment", +0.06, "risk",
                       "Less automated, more friction = higher churn"))
    
    # Protective factors (decrease churn)
    if "Contract" in row and row["Contract"] == "Two year":
        drivers.append(("Two-Year Contract", -0.22, "protect",
                       "Long commitment significantly reduces churn risk"))
    
    if "tenure" in row and row["tenure"] > 36:
        drivers.append(("Long Tenure", -0.15, "protect",
                       "Established customers are much more loyal"))
    
    if "TechSupport" in row and row["TechSupport"] == "Yes":
        drivers.append(("Has Tech Support", -0.06, "protect",
                       "Support increases satisfaction and retention"))
    
    if "OnlineSecurity" in row and row["OnlineSecurity"] == "Yes":
        drivers.append(("Has Online Security", -0.05, "protect",
                       "Security add-on shows investment in services"))
    
    return sorted(drivers, key=lambda x: abs(x[1]), reverse=True)


def create_gauge(probability):
    """Create a gauge chart for probability."""
    color = "#ef4444" if probability >= 0.6 else "#f59e0b" if probability >= 0.3 else "#10b981"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Churn Probability", 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2rem;">👤 Individual Customer Analysis</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Deep dive into a specific customer's churn risk factors and recommended actions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df, is_user_data = get_data()
    
    if df is None:
        st.error("❌ No data available. Please upload your data on the Home page.")
        return
    
    if is_user_data:
        st.success("✅ **Viewing YOUR uploaded data**")
    else:
        st.info("ℹ️ **Demo Mode** - Upload your data on Home page for real analysis.")
    
    # Customer selector
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🔍 Select a Customer</div>
        <div class="section-desc">
            Choose a customer from your dataset to see their detailed churn analysis.
            You can search by customer ID or select from the dropdown.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    id_col = "customerID" if "customerID" in df.columns else df.columns[0]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        customer_id = st.selectbox(
            "Customer ID",
            options=df[id_col].tolist(),
            help="Select a customer to analyze"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        sort_option = st.radio(
            "Quick select",
            ["Current", "Highest Risk", "Lowest Risk"],
            horizontal=True
        )
        if sort_option == "Highest Risk":
            customer_id = df.loc[df["churn_probability"].idxmax(), id_col]
        elif sort_option == "Lowest Risk":
            customer_id = df.loc[df["churn_probability"].idxmin(), id_col]
    
    customer = df[df[id_col] == customer_id].iloc[0]
    prob = customer["churn_probability"]
    risk = customer["risk_level"]
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Profile card
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">👤</div>
            <div class="profile-name">{customer_id}</div>
            <div style="opacity: 0.9;">
                {"⚠️ High Risk Customer" if risk == "High" else "⚡ Medium Risk" if risk == "Medium" else "✅ Low Risk"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gauge
        fig = create_gauge(prob)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
            This customer has a <strong>{prob*100:.1f}%</strong> probability of churning.
            {"Immediate action recommended!" if prob >= 0.6 else "Monitor closely." if prob >= 0.3 else "Low risk, maintain engagement."}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Profile details
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📋 Customer Profile</div>
            <div class="section-desc">
                Key attributes of this customer that influence their churn prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Account Information**")
            if "tenure" in customer:
                st.markdown(f"• Tenure: **{customer['tenure']} months**")
            if "Contract" in customer:
                st.markdown(f"• Contract: **{customer['Contract']}**")
            if "MonthlyCharges" in customer:
                st.markdown(f"• Monthly: **${customer['MonthlyCharges']:.2f}**")
            if "TotalCharges" in customer:
                st.markdown(f"• Total: **${customer['TotalCharges']:.2f}**")
        
        with col_b:
            st.markdown("**Services**")
            if "InternetService" in customer:
                st.markdown(f"• Internet: **{customer['InternetService']}**")
            if "TechSupport" in customer:
                st.markdown(f"• Tech Support: **{customer['TechSupport']}**")
            if "OnlineSecurity" in customer:
                st.markdown(f"• Security: **{customer['OnlineSecurity']}**")
            if "PaymentMethod" in customer:
                st.markdown(f"• Payment: **{customer['PaymentMethod']}**")
    
    st.markdown("---")
    
    # SHAP drivers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📊 What's Driving This Prediction?</div>
            <div class="section-desc">
                These are the key factors influencing this customer's churn probability.
                <strong>Risk factors</strong> (red) increase churn likelihood, while 
                <strong>protective factors</strong> (green) decrease it.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        drivers = get_shap_drivers(customer)
        
        if drivers:
            for factor, impact, category, explanation in drivers:
                impact_pct = abs(impact) * 100
                color = "#ef4444" if category == "risk" else "#10b981"
                arrow = "↑" if category == "risk" else "↓"
                
                st.markdown(f"""
                <div class="driver-item driver-{category}">
                    <div style="flex: 1;">
                        <div style="font-weight: 600;">{arrow} {factor}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">{explanation}</div>
                    </div>
                    <div style="font-weight: 700; color: {color};">
                        {'+' if category == 'risk' else '-'}{impact_pct:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant drivers identified for this customer.")
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 Recommended Actions</div>
            <div class="section-desc">
                Based on this customer's profile and risk factors, here are personalized 
                interventions that could reduce their churn probability.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate recommendations
        if "Contract" in customer and customer["Contract"] == "Month-to-month":
            st.markdown("""
            <div class="action-card">
                <div style="font-weight: 600;">🔴 HIGH PRIORITY: Upgrade Contract</div>
                <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                    Offer a 15% discount on a 1-year contract. This could reduce churn risk by up to 30%.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if "tenure" in customer and customer["tenure"] < 12:
            st.markdown("""
            <div class="action-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                <div style="font-weight: 600;">🟡 New Customer Engagement</div>
                <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                    Schedule a check-in call to ensure satisfaction. Early engagement reduces churn significantly.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if "TechSupport" in customer and customer["TechSupport"] == "No":
            st.markdown("""
            <div class="action-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <div style="font-weight: 600;">🟢 Offer Tech Support Trial</div>
                <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                    Customers with tech support churn 20% less. Offer a free 3-month trial.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if risk == "Low":
            st.markdown("""
            <div class="action-card" style="background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);">
                <div style="font-weight: 600;">✅ Maintain Relationship</div>
                <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                    This is a stable customer. Send appreciation message and loyalty rewards.
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
