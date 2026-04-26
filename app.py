import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data_loader import load_all_regions
from src.train_model import train_and_evaluate
from src.config import TOTAL_BUDGET_USD, TOTAL_WELLS_CHOSEN, REVENUE_PER_UNIT, RISK_THRESHOLD

# Page config and CSS
st.set_page_config(
    page_title="OilyGiant | Oil Well Site Selector",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    div[data-testid="stMetric"] { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #1f77b4; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.05); 
    }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)


COST_PER_WELL    = TOTAL_BUDGET_USD / TOTAL_WELLS_CHOSEN
BREAKEVEN_VOL    = COST_PER_WELL / REVENUE_PER_UNIT  


# Data and Bootstrap processing(cached)
@st.cache_data
def get_real_analysis():
    # Loads real CSVs, trains models, and runs 1000 bootstrap iterations.
    regions_dict = load_all_regions()
    results = {}
    
    rng = np.random.RandomState(54321)
    
    for name, df in regions_dict.items():
        clean_name = name.replace("_", " ").title()
        
        #Train model
        model, y_val, preds, rmse = train_and_evaluate(df)
        
        profits = []
        #Bootstrap 1000 times
        for _ in range(1000):
            sample_idx = y_val.sample(n=500, replace=True, random_state=rng).index
            t_s = y_val.loc[sample_idx]
            p_s = preds.loc[sample_idx]
            top200 = p_s.nlargest(200).index
            revenue = t_s.loc[top200].sum() * REVENUE_PER_UNIT
            profits.append(revenue - TOTAL_BUDGET_USD)
            
        profits = np.array(profits)
        
        results[clean_name] = {
            "avg_pred_vol": preds.mean(),
            "rmse": rmse,
            "mean_profit": profits.mean(),
            "ci_low": np.percentile(profits, 2.5),
            "ci_high": np.percentile(profits, 97.5),
            "risk": (profits < 0).mean(),
            "distribution": profits
        }
    return results

# Load data
with st.spinner("Loading models and running Bootstrapping simulations..."):
    analysis_data = get_real_analysis()


# SIDEBAR NAVIGATION
with st.sidebar:
    st.image("blackgold-price.png", width=80)
    st.title("OilyGiant Analytics")
    st.markdown("Use the navigation below to explore the project.")
    
    nav_option = st.radio("Navigation", [
        "1. Business Problem", 
        "2. Break-Even & Model", 
        "3. Risk Distribution (Bootstrap)", 
        "4. Final Recommendation"
    ])    
footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa; 
        color: #6c757d;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        border-top: 1px solid #e9ecef;
        z-index: 999;
    }

    .block-container {
        padding-bottom: 80px; 
    }
    </style>
    <div class="footer">
        Developed by <b>Leviton Lima Carvalho</b> as a portfolio project | Model trained on Beta Bank historical data
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)

# MAIN CONTENT
st.title("OilyGiant — Selecting the Best Oil Well Location")

#Problem Statement
if nav_option == "1. Business Problem":
    st.header("1. The Business Problem")
    st.markdown(f"""
    OilyGiant has geological survey data for **three candidate regions** and a fixed capital budget of
    **USD ${TOTAL_BUDGET_USD/1e6:.0f} Million** — enough to drill exactly **{TOTAL_WELLS_CHOSEN} wells**.

    The challenge is **not** simply to find the region with the highest average oil volume.
    Random drilling would almost certainly lead to financial losses, because the average well in every
    region falls *below* the break-even threshold.

    The real challenge is threefold:
    1. **Predict** which specific wells will be highly productive, using Machine Learning.
    2. **Estimate** the expected profit and its statistical distribution, using Bootstrapping.
    3. **Select** only the region where the probability of financial loss is below **2.5%**.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Budget", f"${TOTAL_BUDGET_USD/1e6:.0f} M")
    col2.metric("Wells to Drill", f"{TOTAL_WELLS_CHOSEN}")
    col3.metric("Max Risk of Loss", "2.5 %")

#Break-even and model
elif nav_option == "2. Break-Even & Model":
    st.header("2. Break-Even Analysis: Why Random Drilling Fails")
    st.markdown(f"""
    Before any modelling, we calculate the minimum oil volume a single well must produce to simply
    **cover its own drilling cost**.
    
    Break-even = **{BREAKEVEN_VOL:.2f} thousand barrels** per well.
    """)

    region_avgs = {r: data["avg_pred_vol"] for r, data in analysis_data.items()}

    fig_be, ax_be = plt.subplots(figsize=(8, 3))
    fig_be.patch.set_facecolor('none')
    ax_be.set_facecolor('none')
    
    bars = ax_be.bar(region_avgs.keys(), region_avgs.values(), color="#d62728", alpha=0.8, width=0.4)
    ax_be.axhline(BREAKEVEN_VOL, color="#2c3e50", linestyle="--", linewidth=2, label=f"Break-even: {BREAKEVEN_VOL:.1f} k bbl")
    for bar, val in zip(bars, region_avgs.values()):
        ax_be.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}k", ha="center", fontsize=10, color="#2c3e50")
    
    ax_be.set_ylabel("Average Volume (k bbl)", color="#2c3e50")
    ax_be.tick_params(colors='#2c3e50')
    for spine in ax_be.spines.values():
        spine.set_color('#bdc3c7')
        
    ax_be.set_ylim(0, 140)
    ax_be.legend()
    st.pyplot(fig_be)

    st.markdown("""
    **Every region falls below the break-even line.** A Machine Learning model is indispensable to identify the specific subset of wells that exceed this threshold.
    
    ---
    ### Linear Regression — Model Performance
    A Linear Regression model was trained. The key metric is **RMSE** (lower is better).
    """)

    col1, col2, col3 = st.columns(3)
    for i, (reg, data) in enumerate(analysis_data.items()):
        with [col1, col2, col3][i]:
            st.metric(f"{reg} RMSE", f"{data['rmse']:.2f} k bbl")
            if data['rmse'] < 5:
                st.success("✅ Near-perfect predictions")
            else:
                st.error("❌ High error — unreliable")

#Bootstrap Distribution
elif nav_option == "3. Risk Distribution (Bootstrap)":
    st.header("3. Bootstrapping — Profit Distribution (1,000 Scenarios)")
    st.markdown("""
    The chart below is the **centrepiece of this analysis**. It represents the profit outcomes across 1,000 simulated drilling campaigns (surveying 500 wells, drilling the best 200).
    The further a curve extends to the **right of $0**, the safer and more profitable that region is.
    """)

    col_cb1, col_cb2, col_cb3 = st.columns(3)
    show_r0 = col_cb1.checkbox("Show Region 0", value=True)
    show_r1 = col_cb2.checkbox("Show Region 1", value=True)
    show_r2 = col_cb3.checkbox("Show Region 2", value=True)

    region_colors = {"Region 0": "#1f77b4", "Region 1": "#2ca02c", "Region 2": "#ff7f0e"}
    show_flags = {"Region 0": show_r0, "Region 1": show_r1, "Region 2": show_r2}

    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    fig_dist.patch.set_facecolor('none')
    ax_dist.set_facecolor('none')
    
    for name, show in show_flags.items():
        if show and name in analysis_data:
            dist = analysis_data[name]["distribution"]
            ax_dist.hist(dist / 1e6, bins=50, alpha=0.6, color=region_colors[name], label=name, edgecolor="white")

    ax_dist.axvline(0, color="#d62728", linewidth=2.5, linestyle="--", label="Break-even ($0 profit)")
    ax_dist.set_xlabel("Profit per Campaign (USD millions)", fontsize=11, color="#2c3e50")
    ax_dist.set_ylabel("Frequency", fontsize=11, color="#2c3e50")
    ax_dist.tick_params(colors='#2c3e50')
    
    for spine in ax_dist.spines.values():
        spine.set_color('#bdc3c7')
        
    ax_dist.legend()
    st.pyplot(fig_dist)
    
    st.info("""
    **How to read this chart:**
    - **Region 1 (Green)** is almost entirely to the right of zero, indicating consistent profitability.
    - **Regions 0 and 2** have long tails crossing the red line, representing significant simulated losses.
    """)

#Final Recommendation
elif nav_option == "4. Final Recommendation":
    st.header("4. Final Recommendation & Business Decision")
    
    st.success("""
    ### ✅ RECOMMENDED REGION: **Region 1**
    """)
    
    r1_data = analysis_data.get("Region 1", analysis_data[list(analysis_data.keys())[1]])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Expected Mean Profit", f"${r1_data['mean_profit']:,.2f}")
        st.metric("95% Confidence (Lower Bound)", f"${r1_data['ci_low']:,.2f}")
    with col2:
        st.metric("Risk of Financial Loss", f"{r1_data['risk']*100:.2f}%")
        st.metric("Model RMSE", f"{r1_data['rmse']:.2f}")

    st.markdown("""
    **Summary of the decision logic:**
    1. **Break-even analysis** proved random drilling is not viable.
    2. **Model performance (RMSE)** revealed the linear model is highly precise *only* for Region 1.
    3. **Bootstrapping** confirmed Region 1 is the only region where the probability of loss is below the 2.5% tolerance threshold.
    4. Region 1 also delivers the highest expected profit, making it the dominant choice on *both* safety and return.
    """)