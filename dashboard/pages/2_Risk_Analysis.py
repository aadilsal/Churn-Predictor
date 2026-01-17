"""
Page 2: Risk Analysis
=====================

Dynamic risk analysis with user data and full descriptions.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Risk Analysis", page_icon="⚠️", layout="wide")

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    .customer-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 5px solid;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .customer-card:hover {
        transform: translateX(4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .high-risk { border-left-color: #ef4444; }
    .medium-risk { border-left-color: #f59e0b; }
    .low-risk { border-left-color: #10b981; }
    
    .prob-bar {
        height: 8px;
        border-radius: 4px;
        background: #e5e7eb;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .action-chip {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem;
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
</style>
""", unsafe_allow_html=True)


def get_data():
    """Get data from session state or load sample."""
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
        if "MonthlyCharges" in df.columns:
            df["annual_value"] = df["MonthlyCharges"] * 12
        return df, False
    return None, False


def get_recommendations(row):
    """Generate personalized recommendations."""
    recs = []
    
    if "Contract" in row and row["Contract"] == "Month-to-month":
        recs.append(("HIGH", "Offer Annual Contract", "Could reduce risk by 30%"))
    
    if "tenure" in row and row["tenure"] < 12:
        recs.append(("HIGH", "Enhanced Onboarding", "New customer needs engagement"))
    
    if "TechSupport" in row and row["TechSupport"] == "No":
        recs.append(("MEDIUM", "Free Tech Support Trial", "Support users churn 20% less"))
    
    if "PaymentMethod" in row and row["PaymentMethod"] == "Electronic check":
        recs.append(("MEDIUM", "Auto-Pay Incentive", "Reduces churn by 15%"))
    
    if not recs:
        recs.append(("LOW", "Maintain Engagement", "Send appreciation message"))
    
    return recs[:3]


def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2rem;">⚠️ Risk Analysis Dashboard</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Identify and prioritize at-risk customers for targeted retention interventions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df, is_user_data = get_data()
    
    if df is None:
        st.error("❌ No data available. Please upload your data on the Home page.")
        return
    
    if is_user_data:
        st.success("✅ **Analyzing YOUR uploaded data**")
    else:
        st.warning("⚠️ **Demo Mode** - Upload your data on Home page for real analysis.")
    
    # Explanation
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🎯 What This Page Shows</div>
        <div class="section-desc">
            This page lists customers sorted by their <strong>churn probability</strong> (highest risk first).
            For each customer, you'll see:
            <br>• <strong>Churn Probability</strong> - The likelihood (0-100%) that this customer will cancel
            <br>• <strong>Risk Level</strong> - High (>60%), Medium (30-60%), or Low (<30%)
            <br>• <strong>Recommended Actions</strong> - Personalized interventions based on their profile
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Filters")
    
    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        options=["High", "Medium", "Low"],
        default=["High", "Medium"],
        help="Select which risk levels to display"
    )
    
    if "Contract" in df.columns:
        contract_filter = st.sidebar.multiselect(
            "Contract Type",
            options=df["Contract"].unique().tolist(),
            default=df["Contract"].unique().tolist(),
        )
        df = df[df["Contract"].isin(contract_filter)]
    
    min_prob = st.sidebar.slider(
        "Minimum Probability",
        0, 100, 30,
        help="Only show customers above this probability"
    )
    
    # Apply filters
    filtered = df[
        (df["risk_level"].isin(risk_filter)) &
        (df["churn_probability"] >= min_prob / 100)
    ].sort_values("churn_probability", ascending=False)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Customers Shown",
            f"{len(filtered):,}",
            help="Number of customers matching your filters"
        )
    
    with col2:
        if "annual_value" in filtered.columns:
            total_value = filtered["annual_value"].sum()
            st.metric(
                "Total Annual Value",
                f"${total_value:,.0f}",
                help="Combined annual revenue from filtered customers"
            )
        else:
            st.metric("High Risk", f"{(filtered['risk_level'] == 'High').sum():,}")
    
    with col3:
        avg_prob = filtered["churn_probability"].mean() * 100
        st.metric(
            "Avg Churn Probability",
            f"{avg_prob:.1f}%",
            help="Average probability across filtered customers"
        )
    
    with col4:
        if "annual_value" in filtered.columns:
            risk_value = (filtered["annual_value"] * filtered["churn_probability"]).sum()
            st.metric(
                "Revenue at Risk",
                f"${risk_value:,.0f}",
                help="Expected revenue loss based on churn probabilities"
            )
    
    st.markdown("---")
    
    # Customer list
    st.markdown(f"### 👥 At-Risk Customer List ({len(filtered):,} customers)")
    st.markdown("*Sorted by churn probability (highest first). Click to expand for recommendations.*")
    
    # Pagination
    items_per_page = 20
    total_pages = max(1, (len(filtered) - 1) // items_per_page + 1)
    page = st.selectbox("Page", range(1, total_pages + 1), format_func=lambda x: f"Page {x} of {total_pages}")
    
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_data = filtered.iloc[start_idx:end_idx]
    
    for idx, row in page_data.iterrows():
        prob = row["churn_probability"]
        risk = row["risk_level"]
        prob_color = "#ef4444" if risk == "High" else "#f59e0b" if risk == "Medium" else "#10b981"
        risk_class = "high-risk" if risk == "High" else "medium-risk" if risk == "Medium" else "low-risk"
        
        customer_id = row.get("customerID", f"Customer {idx}")
        
        with st.expander(f"**{customer_id}** — {prob*100:.1f}% probability | {risk} Risk", expanded=(idx < start_idx + 3)):
            col1, col2, col3 = st.columns([1.5, 1.5, 2])
            
            with col1:
                st.markdown("##### 📊 Risk Assessment")
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-weight: 600;">Churn Probability</span>
                        <span style="color: {prob_color}; font-weight: 700;">{prob*100:.1f}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: {prob*100}%; background: {prob_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Risk Level:** {risk}")
                if "tenure" in row:
                    st.markdown(f"**Tenure:** {row['tenure']} months")
                if "MonthlyCharges" in row:
                    st.markdown(f"**Monthly Revenue:** ${row['MonthlyCharges']:.2f}")
            
            with col2:
                st.markdown("##### 📋 Customer Profile")
                if "Contract" in row:
                    st.markdown(f"**Contract:** {row['Contract']}")
                if "InternetService" in row:
                    st.markdown(f"**Internet:** {row['InternetService']}")
                if "PaymentMethod" in row:
                    st.markdown(f"**Payment:** {row['PaymentMethod']}")
                if "TechSupport" in row:
                    st.markdown(f"**Tech Support:** {row['TechSupport']}")
            
            with col3:
                st.markdown("##### 🎯 Recommended Actions")
                recommendations = get_recommendations(row)
                for priority, action, reason in recommendations:
                    emoji = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
                    st.markdown(f"""
                    {emoji} **{action}**  
                    <span style="color: #64748b; font-size: 0.85rem;">{reason}</span>
                    """, unsafe_allow_html=True)
    
    # Export
    st.markdown("---")
    st.markdown("### 📥 Export Risk Report")
    st.markdown("Download the filtered customer list with risk scores and recommendations.")
    
    export_df = filtered[["customerID", "churn_probability", "risk_level"]].copy() if "customerID" in filtered.columns else filtered[["churn_probability", "risk_level"]].copy()
    export_df["churn_probability"] = (export_df["churn_probability"] * 100).round(1)
    export_df.columns = ["Customer ID", "Churn Probability (%)", "Risk Level"] if "customerID" in filtered.columns else ["Churn Probability (%)", "Risk Level"]
    
    csv = export_df.to_csv(index=False)
    st.download_button(
        "📥 Download Risk Report (CSV)",
        csv,
        "churn_risk_report.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()
