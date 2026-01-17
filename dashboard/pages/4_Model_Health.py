"""
Page 4: Model Health & Trust
============================

Model reliability, drift monitoring, and trust indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Model Health", page_icon="🔧", layout="wide")

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    .health-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .health-good {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
    }
    
    .health-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
    }
    
    .health-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
    }
    
    .metric-box {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
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
    
    .trust-indicator {
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .trust-reliable {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
    }
    
    .trust-caution {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .trust-warning {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)


def generate_mock_data():
    """Generate mock monitoring data for demo."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    
    drift_data = pd.DataFrame({
        "date": dates,
        "drift_score": np.clip(np.random.normal(0.15, 0.08, 30), 0, 0.5),
        "prediction_shift": np.clip(np.random.normal(0.05, 0.03, 30), 0, 0.2),
    })
    
    perf_data = pd.DataFrame({
        "date": pd.date_range(end=datetime.now(), periods=12, freq="W"),
        "roc_auc": 0.84 + np.random.normal(0, 0.02, 12),
        "accuracy": 0.74 + np.random.normal(0, 0.02, 12),
    })
    
    return drift_data, perf_data


def main():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2rem;">🔧 Model Health & Trust</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Monitor model reliability, detect drift, and understand when to trust predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📊 What This Page Shows</div>
        <div class="section-desc">
            Machine learning models can degrade over time as the real world changes. This page monitors:
            <br><br>
            <strong>• Data Drift</strong> - Are incoming customers different from training data?<br>
            <strong>• Model Performance</strong> - Is the model still accurate?<br>
            <strong>• Trust Indicators</strong> - When should you trust the predictions?
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load monitoring data
    drift_data, perf_data = generate_mock_data()
    
    latest_drift = drift_data.iloc[-1]["drift_score"]
    latest_shift = drift_data.iloc[-1]["prediction_shift"]
    latest_roc = perf_data.iloc[-1]["roc_auc"]
    
    # Overall health status
    if latest_drift < 0.25 and latest_roc > 0.80:
        health_status = "healthy"
        health_class = "health-good"
        health_text = "✅ Model is Healthy"
        health_desc = "All metrics within acceptable ranges. Predictions are reliable."
    elif latest_drift < 0.40 and latest_roc > 0.75:
        health_status = "caution"
        health_class = "health-warning"
        health_text = "⚠️ Caution Advised"
        health_desc = "Some drift detected. Monitor predictions closely."
    else:
        health_status = "critical"
        health_class = "health-critical"
        health_text = "🔴 Attention Required"
        health_desc = "Significant drift or performance decline. Consider retraining."
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="health-badge {health_class}">{health_text}</span>
        <p style="color: #64748b; margin-top: 0.5rem;">{health_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{latest_drift*100:.1f}%</div>
            <div class="metric-label">Data Drift Score</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("📊 % of features that have drifted from training data")
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{latest_shift*100:.1f}%</div>
            <div class="metric-label">Prediction Shift</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("📈 Change in average prediction from baseline")
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{latest_roc:.3f}</div>
            <div class="metric-label">ROC-AUC Score</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("🎯 Model discrimination ability (0.5 = random, 1.0 = perfect)")
    
    with col4:
        days_since = 7  # Mock
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{days_since} days</div>
            <div class="metric-label">Since Last Retrain</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("🔄 Time since model was last updated")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📉 Data Drift Over Time</div>
            <div class="section-desc">
                Tracks how different current customer data is from the training data.
                <strong>When drift exceeds 50%</strong>, the model may become unreliable.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drift_data["date"],
            y=drift_data["drift_score"] * 100,
            mode="lines+markers",
            name="Drift Score",
            line=dict(color="#667eea", width=3),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.1)",
        ))
        
        fig.add_hline(y=50, line_dash="dash", line_color="#ef4444",
                     annotation_text="Alert Threshold (50%)")
        fig.add_hline(y=25, line_dash="dash", line_color="#f59e0b",
                     annotation_text="Warning (25%)")
        
        fig.update_layout(
            yaxis_title="Drift Score (%)",
            height=350,
            yaxis=dict(range=[0, 60]),
            font=dict(family="Inter"),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📈 Model Performance Trend</div>
            <div class="section-desc">
                Tracks ROC-AUC score over time. This measures how well the model 
                distinguishes churners from non-churners.
                <strong>Baseline: 0.84</strong> | <strong>Minimum acceptable: 0.72</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=perf_data["date"],
            y=perf_data["roc_auc"],
            mode="lines+markers",
            name="ROC-AUC",
            line=dict(color="#10b981", width=3),
            marker=dict(size=8),
        ))
        
        fig.add_hline(y=0.84, line_dash="dash", line_color="#6b7280",
                     annotation_text="Baseline (0.84)")
        fig.add_hline(y=0.72, line_dash="dash", line_color="#ef4444",
                     annotation_text="Minimum (0.72)")
        
        fig.update_layout(
            yaxis_title="ROC-AUC Score",
            height=350,
            yaxis=dict(range=[0.65, 0.95]),
            font=dict(family="Inter"),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Trust indicators
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🛡️ When to Trust Predictions</div>
        <div class="section-desc">
            Not all predictions are equally reliable. Use these indicators to understand
            when predictions are trustworthy and when to exercise caution.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="trust-indicator trust-reliable">
            <h4 style="margin: 0; color: #065f46;">✅ High Confidence</h4>
            <p style="margin: 0.5rem 0 0 0; color: #047857;">
                Predictions are reliable when:
            </p>
            <ul style="margin: 0.5rem 0 0 0; color: #047857;">
                <li>Drift score < 25%</li>
                <li>ROC-AUC > 0.80</li>
                <li>Customer profile is typical</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="trust-indicator trust-caution">
            <h4 style="margin: 0; color: #92400e;">⚠️ Use with Caution</h4>
            <p style="margin: 0.5rem 0 0 0; color: #b45309;">
                Exercise judgment when:
            </p>
            <ul style="margin: 0.5rem 0 0 0; color: #b45309;">
                <li>Drift score 25-50%</li>
                <li>ROC-AUC 0.72-0.80</li>
                <li>Unusual customer profiles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="trust-indicator trust-warning">
            <h4 style="margin: 0; color: #991b1b;">🔴 Low Confidence</h4>
            <p style="margin: 0.5rem 0 0 0; color: #b91c1c;">
                Seek additional validation when:
            </p>
            <ul style="margin: 0.5rem 0 0 0; color: #b91c1c;">
                <li>Drift score > 50%</li>
                <li>ROC-AUC < 0.72</li>
                <li>Model retraining needed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What the metrics mean
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📚 Glossary: Understanding These Metrics</div>
        <div class="section-desc">
            Here's what each metric means in plain language.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 Data Drift Score"):
        st.markdown("""
        **What it is:** A measure of how different your current customers are from the 
        customers the model was trained on.
        
        **Why it matters:** Models learn patterns from training data. If new customers 
        are very different, those patterns might not apply anymore.
        
        **What to do:** If drift exceeds 50%, consider retraining the model with recent data.
        """)
    
    with st.expander("📈 ROC-AUC Score"):
        st.markdown("""
        **What it is:** A score from 0.5 to 1.0 measuring how well the model separates 
        churners from non-churners.
        
        **Interpretation:**
        - **0.50** = Random guessing (useless)
        - **0.70** = Acceptable
        - **0.80** = Good
        - **0.90+** = Excellent
        
        **Our baseline:** 0.84 (Good)
        """)
    
    with st.expander("🔄 Retraining Triggers"):
        st.markdown("""
        **When to retrain the model:**
        - Data drift > 50%
        - ROC-AUC drops below 0.72
        - Business rules change significantly
        - New customer segments emerge
        
        **Our safeguards:**
        - Maximum 4 retrains per month
        - Minimum 7 days between retrains
        - Requires human approval
        """)


if __name__ == "__main__":
    main()
