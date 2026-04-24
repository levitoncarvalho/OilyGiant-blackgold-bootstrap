import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import load_all_regions
from src.train_model import train_and_evaluate
from src.bootstrap import evaluate_risk

st.set_page_config(page_title="OilyGiant – Oil Well Selection", layout="wide")
st.title("🛢️ OilyGiant – New Well Location Selector")
st.markdown("""
This tool uses **Linear Regression** and **Bootstrapping** to evaluate three potential regions
and recommends the one with the highest expected profit while keeping the **risk of loss below 2.5%**.
""")

@st.cache_data
def get_results():
    regions = load_all_regions()
    results = {}
    for name, df in regions.items():
        _, y_val, preds, rmse = train_and_evaluate(df)
        risk = evaluate_risk(y_val, preds)
        results[name] = {'rmse': rmse, **risk}
    return results

results = get_results()

# Summary table
st.subheader("📊 Regional Summary")
summary_data = []
for name, res in results.items():
    summary_data.append({
        'Region': name.replace('_', ' ').title(),
        'RMSE (k barrels)': f"{res['rmse']:,.2f}",
        'Mean Profit (USD)': f"${res['mean_profit']:,.0f}",
        'Risk of Loss': f"{res['risk_of_loss']:.1%}",
        'Approved': '✅' if res['is_approved'] else '❌'
    })
st.table(pd.DataFrame(summary_data))

# Recommendation
approved = {k: v for k, v in results.items() if v['is_approved']}
if approved:
    best_region = max(approved, key=lambda k: approved[k]['mean_profit'])
    st.success(f"🏆 **Recommended Region:** {best_region.replace('_',' ').title()}")
else:
    st.error("No region meets the 2.5% risk threshold.")

# Profit distributions
st.subheader("📈 Profit Distributions (1,000 Bootstrap Samples)")
fig, ax = plt.subplots(figsize=(10, 5))
for name, res in results.items():
    ax.hist(res['distribution'] / 1e6, bins=50, alpha=0.5,
            label=name.replace('_', ' ').title())
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break‑even')
ax.set_xlabel('Profit (millions USD)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ for OilyGiant | Synthetic data – academic project")