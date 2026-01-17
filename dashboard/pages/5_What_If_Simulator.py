"""
Page 5: What-If Simulator
=========================

Interactive scenario testing for interventions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="What-If Simulator", page_icon="🔮", layout="wide")

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    .scenario-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-top: 4px solid;
    }
    
    .scenario-a { border-top-color: #6366f1; }
    .scenario-b { border-top-color: #10b981; }
    
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: 700;
    }
    
    .result-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .impact-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .impact-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .impact-neutral {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
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


def calculate_probability(contract, tenure, internet, payment, tech_support, security):
    """Calculate churn probability based on features."""
    base_prob = 0.27
    
    # Contract impact (biggest factor)
    if contract == "Month-to-month":
        base_prob += 0.20
    elif contract == "One year":
        base_prob -= 0.08
    else:  # Two year
        base_prob -= 0.18
    
    # Tenure impact
    if tenure < 12:
        base_prob += 0.12
    elif tenure > 36:
        base_prob -= 0.10
    
    # Internet
    if internet == "Fiber optic":
        base_prob += 0.08
    elif internet == "No":
        base_prob -= 0.05
    
    # Services
    if tech_support == "Yes":
        base_prob -= 0.06
    if security == "Yes":
        base_prob -= 0.05
    
    # Payment
    if payment == "Electronic check":
        base_prob += 0.05
    elif payment in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        base_prob -= 0.03
    
    return np.clip(base_prob, 0.05, 0.95)


def main():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2rem;">🔮 What-If Simulator</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Test how different interventions could reduce customer churn risk
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🎯 How This Works</div>
        <div class="section-desc">
            This simulator lets you test hypothetical scenarios to see how changes to a customer's 
            profile would affect their churn probability. 
            <br><br>
            <strong>Use Case Examples:</strong><br>
            • What if we upgrade a customer from month-to-month to annual contract?<br>
            • What if we add tech support to a high-risk customer?<br>
            • How much would churn risk decrease with payment method change?
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two scenarios side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="scenario-card scenario-a">
            <h3 style="margin: 0 0 1rem 0; color: #6366f1;">📊 Scenario A: Current State</h3>
            <p style="color: #64748b; margin-bottom: 1rem;">
                Configure the customer's current situation
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Customer Profile")
        
        contract_a = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            key="contract_a",
            help="The customer's current contract agreement"
        )
        
        tenure_a = st.slider(
            "Tenure (months)",
            1, 72, 6,
            key="tenure_a",
            help="How long the customer has been with you"
        )
        
        internet_a = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"],
            key="internet_a",
            help="The type of internet service"
        )
        
        payment_a = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="payment_a",
            help="How the customer pays their bill"
        )
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tech_a = st.selectbox("Tech Support", ["No", "Yes"], key="tech_a")
        with col_a2:
            security_a = st.selectbox("Online Security", ["No", "Yes"], key="security_a")
        
        prob_a = calculate_probability(contract_a, tenure_a, internet_a, payment_a, tech_a, security_a)
    
    with col2:
        st.markdown("""
        <div class="scenario-card scenario-b">
            <h3 style="margin: 0 0 1rem 0; color: #10b981;">🔧 Scenario B: After Intervention</h3>
            <p style="color: #64748b; margin-bottom: 1rem;">
                Apply interventions to see potential impact
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Customer Profile (Modified)")
        
        contract_b = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            index=1,
            key="contract_b",
            help="Potential new contract type"
        )
        
        tenure_b = st.slider(
            "Tenure (months)",
            1, 72, tenure_a,
            key="tenure_b",
            help="Tenure can't change instantly, but affects projections"
        )
        
        internet_b = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"],
            key="internet_b",
            help="Potential service change"
        )
        
        payment_b = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=2,
            key="payment_b",
            help="Potential new payment method"
        )
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            tech_b = st.selectbox("Tech Support", ["No", "Yes"], index=1, key="tech_b")
        with col_b2:
            security_b = st.selectbox("Online Security", ["No", "Yes"], index=1, key="security_b")
        
        prob_b = calculate_probability(contract_b, tenure_b, internet_b, payment_b, tech_b, security_b)
    
    st.markdown("---")
    
    # Results
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📈 Simulation Results</div>
        <div class="section-desc">
            Comparison of churn probability before and after the proposed interventions.
            The <strong>Impact</strong> shows how much the intervention would reduce (or increase) churn risk.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    color_a = "#ef4444" if prob_a >= 0.6 else "#f59e0b" if prob_a >= 0.3 else "#10b981"
    color_b = "#ef4444" if prob_b >= 0.6 else "#f59e0b" if prob_b >= 0.3 else "#10b981"
    change = prob_b - prob_a
    
    with col1:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {color_a} 0%, {color_a}dd 100%);">
            <div class="result-value">{prob_a*100:.1f}%</div>
            <div class="result-label">Scenario A (Current)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if change < -0.01:
            impact_class = "impact-positive"
            impact_text = f"↓ {abs(change)*100:.1f}% reduction"
        elif change > 0.01:
            impact_class = "impact-negative"
            impact_text = f"↑ {abs(change)*100:.1f}% increase"
        else:
            impact_class = "impact-neutral"
            impact_text = "No significant change"
        
        st.markdown(f"""
        <div class="result-card {impact_class}">
            <div class="result-value">{change*100:+.1f}%</div>
            <div class="result-label">{impact_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {color_b} 0%, {color_b}dd 100%);">
            <div class="result-value">{prob_b*100:.1f}%</div>
            <div class="result-label">Scenario B (After)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Visual Comparison")
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=["Current", "After Intervention"],
            y=[prob_a * 100, prob_b * 100],
            marker_color=[color_a, color_b],
            text=[f"{prob_a*100:.1f}%", f"{prob_b*100:.1f}%"],
            textposition="outside",
            textfont=dict(size=16, color=["#1e293b", "#1e293b"])
        ))
        
        fig.add_hline(y=60, line_dash="dash", line_color="#ef4444", 
                     annotation_text="High Risk (60%)")
        fig.add_hline(y=30, line_dash="dash", line_color="#f59e0b",
                     annotation_text="Medium Risk (30%)")
        
        fig.update_layout(
            height=350,
            yaxis_title="Churn Probability (%)",
            yaxis=dict(range=[0, max(prob_a, prob_b) * 100 + 20]),
            font=dict(family="Inter"),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Changes Applied")
        
        changes = []
        if contract_a != contract_b:
            changes.append(f"📄 **Contract:** {contract_a} → {contract_b}")
        if payment_a != payment_b:
            changes.append(f"💳 **Payment:** {payment_a.split(' ')[0]}... → {payment_b.split(' ')[0]}...")
        if tech_a != tech_b:
            changes.append(f"🔧 **Tech Support:** {tech_a} → {tech_b}")
        if security_a != security_b:
            changes.append(f"🔒 **Security:** {security_a} → {security_b}")
        if internet_a != internet_b:
            changes.append(f"🌐 **Internet:** {internet_a} → {internet_b}")
        
        if changes:
            for c in changes:
                st.markdown(c)
        else:
            st.info("No changes between scenarios. Modify Scenario B to see the impact.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recommendation
        if change < -0.1:
            st.success(f"""
            ✅ **Strong Recommendation**  
            These changes would reduce churn risk by {abs(change)*100:.1f} percentage points.
            Expected ROI is positive for customers with >$50/month revenue.
            """)
        elif change < -0.01:
            st.info(f"""
            ℹ️ **Moderate Impact**  
            Risk reduction of {abs(change)*100:.1f} points. Consider for high-value customers.
            """)
        elif change > 0.01:
            st.warning(f"""
            ⚠️ **Negative Impact**  
            These changes would increase churn risk. Review your modifications.
            """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; border-left: 4px solid #64748b;">
        <strong style="color: #1e293b;">⚠️ Important Notes</strong><br>
        <ul style="margin: 0.5rem 0 0 0; color: #000000;">
            <li>This simulator uses a simplified model for demonstration</li>
            <li>Actual results may vary based on additional customer factors</li>
            <li>Some changes (like tenure) cannot be applied instantly</li>
            <li>Use these insights as directional guidance, not guarantees</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
