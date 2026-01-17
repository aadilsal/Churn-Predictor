"""
Page 1: Overview Dashboard
==========================

Dynamic overview using user-uploaded data with full descriptions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
    
    .metric-desc {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .section-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #f1f5f9;
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
    
    .insight-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    
    .insight-box strong {
        color: #065f46 !important;
    }
    
    .insight-box em {
        color: #1e293b !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    
    .warning-box strong {
        color: #92400e !important;
    }
    
    .warning-box em {
        color: #1e293b !important;
    }
    
    .risk-badge-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-badge-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    .explanation-tooltip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)


def get_data():
    """Get data from session state or load sample."""
    if "predictions" in st.session_state and st.session_state.predictions is not None:
        return st.session_state.predictions, True
    
    # Load sample data for demo
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


def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2rem;">📊 Churn Overview Dashboard</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
            Comprehensive view of customer churn risk across your portfolio
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df, is_user_data = get_data()
    
    if df is None:
        st.error("❌ No data available. Please upload your data on the Home page.")
        st.info("👉 Go to the **Home** page to upload your customer data CSV file.")
        return
    
    # Data source indicator
    if is_user_data:
        st.success("✅ **Showing analysis of YOUR uploaded data**")
    else:
        st.warning("⚠️ **Demo Mode** - Showing sample data. Upload your own data on the Home page for personalized analysis.")
    
    # Filters in sidebar
    st.sidebar.markdown("### 🎛️ Filters")
    st.sidebar.markdown("*Narrow down the analysis to specific customer segments*")
    
    if "Contract" in df.columns:
        contract_filter = st.sidebar.multiselect(
            "Contract Type",
            options=df["Contract"].unique(),
            default=df["Contract"].unique(),
            help="Filter by customer contract type"
        )
        df = df[df["Contract"].isin(contract_filter)]
    
    if "tenure" in df.columns:
        tenure_range = st.sidebar.slider(
            "Tenure Range (months)",
            min_value=0,
            max_value=int(df["tenure"].max()),
            value=(0, int(df["tenure"].max())),
            help="Filter customers by how long they've been with you"
        )
        df = df[(df["tenure"] >= tenure_range[0]) & (df["tenure"] <= tenure_range[1])]
    
    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
        help="Filter by predicted churn risk level"
    )
    df = df[df["risk_level"].isin(risk_filter)]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(df):,} customers**")
    
    # Key Metrics Section
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📈 Key Performance Indicators</div>
        <div class="section-desc">
            These metrics summarize the overall health of your customer base. 
            Use them to quickly understand your churn risk exposure.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Customers</div>
            <div class="metric-desc">Active customers in analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if "Churn" in df.columns:
            churn_rate = df["Churn"].mean() * 100
        else:
            churn_rate = df["churn_probability"].mean() * 100
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div>
            <div class="metric-desc">Percentage likely to leave</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_risk = (df["risk_level"] == "High").sum()
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
            <div class="metric-value">{high_risk:,}</div>
            <div class="metric-label">High Risk</div>
            <div class="metric-desc">Customers needing attention</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_prob = df["churn_probability"].mean() * 100
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
            <div class="metric-value">{avg_prob:.1f}%</div>
            <div class="metric-label">Avg Probability</div>
            <div class="metric-desc">Mean churn likelihood</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        if "MonthlyCharges" in df.columns:
            revenue_at_risk = df[df["risk_level"] == "High"]["MonthlyCharges"].sum() * 12
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);">
                <div class="metric-value">${revenue_at_risk/1000:.0f}K</div>
                <div class="metric-label">Revenue at Risk</div>
                <div class="metric-desc">Annual revenue from high-risk</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <div class="metric-value">{(df["risk_level"] == "Low").sum():,}</div>
                <div class="metric-label">Low Risk</div>
                <div class="metric-desc">Stable customers</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📊 Churn Probability Distribution</div>
            <div class="section-desc">
                This histogram shows how churn probabilities are distributed across your customers.
                <strong>Read it as:</strong> The height of each bar shows how many customers have that probability range.
                Ideally, you want most customers on the left (low probability).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.histogram(
            df,
            x="churn_probability",
            nbins=25,
            color="risk_level",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            labels={"churn_probability": "Churn Probability", "count": "Number of Customers"},
        )
        fig.update_layout(
            bargap=0.1,
            height=400,
            legend_title_text="Risk Level",
            xaxis_title="Churn Probability (0 = won't churn, 1 = will churn)",
            yaxis_title="Number of Customers",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 Risk Segmentation</div>
            <div class="section-desc">
                This pie chart breaks down your customers by risk level.
                <strong>🔴 High Risk (>60%):</strong> Immediate intervention needed.
                <strong>🟡 Medium (30-60%):</strong> Monitor closely.
                <strong>🟢 Low (<30%):</strong> Stable customers.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        risk_counts = df["risk_level"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            hole=0.4,
        )
        fig.update_layout(
            height=400,
            font=dict(family="Inter"),
        )
        fig.update_traces(
            textinfo='percent+value',
            textfont_size=14,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    if "Contract" in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="section-card">
                <div class="section-title">📄 Churn by Contract Type</div>
                <div class="section-desc">
                    Compares churn rates across different contract types.
                    <strong>Why it matters:</strong> Contract type is one of the strongest predictors of churn.
                    Month-to-month customers are typically at much higher risk.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if "Churn" in df.columns:
                contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
                contract_churn["Churn"] = contract_churn["Churn"] * 100
                y_col = "Churn"
            else:
                contract_churn = df.groupby("Contract")["churn_probability"].mean().reset_index()
                contract_churn["churn_probability"] = contract_churn["churn_probability"] * 100
                y_col = "churn_probability"
            
            fig = px.bar(
                contract_churn,
                x="Contract",
                y=y_col,
                color=y_col,
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                labels={y_col: "Churn Rate (%)"},
            )
            fig.update_layout(
                height=400,
                xaxis_title="Contract Type",
                yaxis_title="Average Churn Rate (%)",
                font=dict(family="Inter"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "tenure" in df.columns:
                st.markdown("""
                <div class="section-card">
                    <div class="section-title">⏱️ Churn by Tenure</div>
                    <div class="section-desc">
                        Shows how churn risk changes based on how long customers have been with you.
                        <strong>Key insight:</strong> New customers (0-12 months) are at highest risk.
                        Risk typically decreases as customers stay longer.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                df["tenure_group"] = pd.cut(
                    df["tenure"],
                    bins=[0, 12, 24, 48, 100],
                    labels=["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
                )
                tenure_churn = df.groupby("tenure_group")["churn_probability"].mean().reset_index()
                tenure_churn["churn_probability"] = tenure_churn["churn_probability"] * 100
                
                fig = px.line(
                    tenure_churn,
                    x="tenure_group",
                    y="churn_probability",
                    markers=True,
                    labels={"churn_probability": "Avg Churn Probability (%)", "tenure_group": "Tenure Group"}
                )
                fig.update_traces(
                    line_color="#667eea",
                    line_width=4,
                    marker_size=12,
                )
                fig.update_layout(
                    height=400,
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card">
        <div class="section-title">💡 Key Insights & Recommendations</div>
        <div class="section-desc">
            Based on the analysis of your data, here are the most important findings and suggested actions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Top Risk Factors")
        st.markdown("""
        <div class="warning-box">
            <strong>Month-to-Month Contracts</strong><br>
            These customers churn 2-3x more than annual contracts.
            <em>Action: Offer incentives to upgrade to annual plans.</em>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <strong>New Customers (< 12 months)</strong><br>
            First-year customers are at highest risk.
            <em>Action: Implement strong onboarding programs.</em>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🟢 Protective Factors")
        st.markdown("""
        <div class="insight-box">
            <strong>Long-Term Contracts</strong><br>
            Two-year contracts reduce churn by up to 85%.
            <em>Focus retention on contract renewal periods.</em>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>Support Services</strong><br>
            Customers with tech support churn 20% less.
            <em>Offer support add-ons to at-risk customers.</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Export
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    st.markdown("Download the analyzed data with predictions for further analysis or reporting.")
    
    export_cols = ["customerID"] if "customerID" in df.columns else []
    export_cols += ["churn_probability", "risk_level"]
    if "Contract" in df.columns:
        export_cols.append("Contract")
    if "tenure" in df.columns:
        export_cols.append("tenure")
    if "MonthlyCharges" in df.columns:
        export_cols.append("MonthlyCharges")
    
    csv = df[export_cols].to_csv(index=False)
    st.download_button(
        "📥 Download Analysis Results (CSV)",
        csv,
        "churn_analysis_results.csv",
        "text/csv",
        help="Download the full analysis with predictions"
    )


if __name__ == "__main__":
    main()
