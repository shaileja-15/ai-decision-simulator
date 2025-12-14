import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================================================================
# MODEL AND FINANCIAL PARAMETERS (HARDCODED FROM PRE-ANALYSIS)
# ================================================================================
# These constants make the Streamlit app fast, self-contained, and deployable
# without needing to retrain models or reload data repeatedly.

# 1. Predictive Model Coefficients (Trained to predict Units_Sold from the CSV)
# Features: [Price, Marketing_Spend, Economic_Index, Seasonality_Index]
MODEL_COEFFICIENTS = [-500.1655890147692, 0.5001327248151396, -19.72212910696324, -4.648546615296999]
MODEL_INTERCEPT = 51982.74464655321

# 2. Financial Ratios (Averages from Historical Data)
RATIOS = {
    'AR_to_Revenue_Ratio': 0.9714, # Accounts Receivable as % of Revenue
    'AP_to_Cost_Ratio': 1.3151,     # Accounts Payable as % of Cost
    'Inventory_per_Unit': 35.78,    # Inventory Value per Unit Sold
    'Depreciation_to_Revenue_Ratio': 0.05 # Assumed Depreciation Rate
}

# 3. Base Case Averages (Historical Mean Values)
BASE_FINANCE = {
    'Avg_Price': 61.34, 
    'Avg_Marketing': 18451.10, 
    'Avg_Economic': 100.86, 
    'Avg_Seasonality': 1.05, 
    'Avg_Unit_Cost': 27.05,
    'Avg_Profit': 1025804.70,
    'Avg_Revenue': 1766932.38,
    'Avg_Total_Cost': 741127.68,
    'Avg_Units': 27515.22,
    'Avg_AR': 1716447.88,
    'Avg_Inventory': 985160.82, 
    'Avg_Current_Liabilities': 982563.18, 
    'Avg_Current_Assets': 7892305.51 
}

# Calculated Base Metrics
BASE_FINANCE['Base_Depreciation'] = BASE_FINANCE['Avg_Revenue'] * RATIOS['Depreciation_to_Revenue_Ratio']
BASE_FINANCE['Base_FCF'] = BASE_FINANCE['Avg_Profit'] + BASE_FINANCE['Base_Depreciation']
BASE_FINANCE['Base_Quick_Ratio'] = (BASE_FINANCE['Avg_Current_Assets'] - BASE_FINANCE['Avg_Inventory']) / BASE_FINANCE['Avg_Current_Liabilities']
SCENARIO_PROFIT_STD_PROXY = BASE_FINANCE['Avg_Profit'] * 0.3 # Proxy for risk estimation
NUM_SIMULATIONS = 5000

# ================================================================================
# CORE PREDICTION & FINANCIAL MODELING FUNCTION
# ================================================================================
def predict_scenario_outcomes(price_chg, marketing_chg):
    """Calculates profit, FCF, Quick Ratio, and runs Monte Carlo for a given scenario."""
    
    # 1. Apply user-defined changes
    new_price = BASE_FINANCE['Avg_Price'] * (1 + price_chg)
    new_marketing = BASE_FINANCE['Avg_Marketing'] * (1 + marketing_chg)
    
    # 2. Predict Units Sold (Demand Curve)
    predicted_units_sold = max(
        0,
        MODEL_INTERCEPT
        + MODEL_COEFFICIENTS[0] * new_price
        + MODEL_COEFFICIENTS[1] * new_marketing
        + MODEL_COEFFICIENTS[2] * BASE_FINANCE['Avg_Economic']
        + MODEL_COEFFICIENTS[3] * BASE_FINANCE['Avg_Seasonality']
    )

    # 3. Calculate Core Financials
    predicted_revenue = new_price * predicted_units_sold
    predicted_total_cost = BASE_FINANCE['Avg_Unit_Cost'] * predicted_units_sold
    predicted_profit = predicted_revenue - predicted_total_cost

    # 4. Model Free Cash Flow (FCF) Drivers
    new_ar = predicted_revenue * RATIOS['AR_to_Revenue_Ratio']
    new_inventory = predicted_units_sold * RATIOS['Inventory_per_Unit']
    new_ap = predicted_total_cost * RATIOS['AP_to_Cost_Ratio']
    
    current_ap = BASE_FINANCE['Avg_Total_Cost'] * RATIOS['AP_to_Cost_Ratio']
    base_nwc = BASE_FINANCE['Avg_AR'] + BASE_FINANCE['Avg_Inventory'] - current_ap
    new_nwc = new_ar + new_inventory - new_ap
    change_in_nwc = new_nwc - base_nwc

    predicted_depreciation = predicted_revenue * RATIOS['Depreciation_to_Revenue_Ratio']
    predicted_fcf = predicted_profit + predicted_depreciation - change_in_nwc
    
    # 5. Calculate New Liquidity (Quick Ratio)
    const_current_assets = BASE_FINANCE['Avg_Current_Assets'] - BASE_FINANCE['Avg_AR'] - BASE_FINANCE['Avg_Inventory']
    const_current_liabilities = BASE_FINANCE['Avg_Current_Liabilities'] - current_ap
    
    new_current_assets = const_current_assets + new_ar + new_inventory
    new_current_liabilities = const_current_liabilities + new_ap
    new_quick_ratio = (new_current_assets - new_inventory) / new_current_liabilities
    
    # 6. Run Monte Carlo Risk Simulation
    np.random.seed(42)
    simulated_profits = np.random.normal(predicted_profit, SCENARIO_PROFIT_STD_PROXY, NUM_SIMULATIONS)
    simulated_profits[simulated_profits < 0] = 0 
    loss_prob = (simulated_profits < 0).mean() * 100
    VaR_95 = np.percentile(simulated_profits, 5)

    return {
        'Profit': predicted_profit,
        'FCF': predicted_fcf,
        'Quick_Ratio': new_quick_ratio,
        'Units_Sold': predicted_units_sold,
        'Loss_Prob': loss_prob,
        'VaR_95': VaR_95,
        'Sim_Profits': simulated_profits
    }

# ================================================================================
# STREAMLIT PAGE CONFIGURATION AND UI
# ================================================================================

# Page configuration
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")

st.title("🔥 AI Dynamic Strategic Decision Simulator")
st.markdown("### Two-Stage Financial Planning, Liquidity Analysis, and Risk Protocol Engine.")


# -------------------------------
# 1. Sidebar: Dual Scenario Input
# -------------------------------
st.sidebar.header("🎯 Define Strategy Scenarios")
st.sidebar.markdown(f"**Base Case (Avg Profit): ${BASE_FINANCE['Avg_Profit']:,.0f}**")
st.sidebar.markdown(f"**Base Case (Quick Ratio): {BASE_FINANCE['Base_Quick_Ratio']:.2f}**")

# Scenario A: Immediate Strategy
st.sidebar.subheader("Stage A: IMMEDIATE Strategy (Current Action)")
price_a = st.sidebar.slider("Price Change A (%)", -30, 30, 0, 1, key='price_a') / 100.0
marketing_a = st.sidebar.slider("Marketing Spend A (%)", -50, 100, 0, 5, key='marketing_a') / 100.0

# Scenario B: Future/Contingency Strategy
st.sidebar.subheader("Stage B: FUTURE/CONTINGENCY Strategy (Next Step)")
price_b = st.sidebar.slider("Price Change B (%)", -30, 30, 5, 1, key='price_b') / 100.0
marketing_b = st.sidebar.slider("Marketing Spend B (%)", -50, 100, 10, 5, key='marketing_b') / 100.0

st.sidebar.markdown("---")
# The Button Trigger (Crucial for user request)
run_button = st.sidebar.button("Run AI Strategic Simulation 🤖")

# -------------------------------
# 2. Main Body Output (Conditional on Button Press)
# -------------------------------

if run_button:
    
    st.subheader("...Running AI Financial Models & 5,000 Monte Carlo Simulations...")
    
    # Execute both scenarios
    result_a = predict_scenario_outcomes(price_a, marketing_a)
    result_b = predict_scenario_outcomes(price_b, marketing_b)
    
    # Thresholds for decision logic
    FCF_INCREASE_THRESHOLD = 0.15 
    QUICK_RATIO_CRITICAL = 1.0 

    fcf_a_change_percent = (result_a['FCF'] - BASE_FINANCE['Base_FCF']) / BASE_FINANCE['Base_FCF']
    fcf_b_change_percent = (result_b['FCF'] - BASE_FINANCE['Base_FCF']) / BASE_FINANCE['Base_FCF']
    
    # ================================================================================
    # COMPARATIVE RESULTS TABLE
    # ================================================================================
    st.markdown("## 📊 Dual-Stage Scenario Comparison: Profit, Cash Flow, and Liquidity")
    
    comparison_data = {
        'Metric': ['Predicted Profit (Mean)', 'Predicted FCF (Mean)', 'Quick Ratio (Liquidity)', 'VaR 95% (Worst-Case Profit)', 'Loss Probability (%)'],
        'Base Case': [
            f"${BASE_FINANCE['Avg_Profit']:,.0f}", 
            f"${BASE_FINANCE['Base_FCF']:,.0f}", 
            f"{BASE_FINANCE['Base_Quick_Ratio']:.2f}", 
            "N/A (Historical)", 
            "N/A (Historical)"
        ],
        'Scenario A (IMMEDIATE)': [
            f"${result_a['Profit']:,.0f}", 
            f"${result_a['FCF']:,.0f} ({fcf_a_change_percent:+.1%} $\Delta$)", 
            f"{result_a['Quick_Ratio']:.2f}",
            f"${result_a['VaR_95']:,.0f}",
            f"{result_a['Loss_Prob']:.2f}%"
        ],
        'Scenario B (CONTINGENCY)': [
            f"${result_b['Profit']:,.0f}", 
            f"${result_b['FCF']:,.0f} ({fcf_b_change_percent:+.1%} $\Delta$)", 
            f"{result_b['Quick_Ratio']:.2f}",
            f"${result_b['VaR_95']:,.0f}",
            f"{result_b['Loss_Prob']:.2f}%"
        ]
    }
    comparison_df = pd.DataFrame(comparison_data).set_index('Metric')
    st.dataframe(comparison_df, use_container_width=True)

    # ================================================================================
    # ADVANCED TWO-STAGE RECOMMENDATION ENGINE
    # ================================================================================
    st.markdown("## 🧠 Final AI Strategic Decision Protocol: Conditional Guidance")
    
    # --- Part 1: Immediate Recommendation (Scenario A) ---
    st.subheader("Phase 1: Immediate Strategy Protocol (Scenario A)")

    if result_a['Quick_Ratio'] < QUICK_RATIO_CRITICAL:
        rec_a = "⛔ **STOP! LIQUIDITY FAILURE RISK.** Scenario A will crash the Quick Ratio (below 1.0). Do not proceed. Focus on increasing price or reducing inventory/marketing costs to improve cash position."
        st.error(rec_a)
    elif fcf_a_change_percent > FCF_INCREASE_THRESHOLD and result_a['Loss_Prob'] < 5:
        rec_a = "🚀 **IMMEDIATE EXECUTION: ALPHA STRATEGY.** Scenario A delivers superior FCF growth and low risk. This strategy validates immediate deployment."
        st.success(rec_a)
    elif result_a['Profit'] > BASE_FINANCE['Avg_Profit'] and result_a['VaR_95'] > 0:
        rec_a = "✅ **CONTROLLED EXECUTION: BETA STRATEGY.** Scenario A is profitable and the worst-case VaR is positive. Proceed, but monitor key financial indicators closely."
        st.info(rec_a)
    else:
        rec_a = "🟡 **HOLD/REFINEMENT NEEDED.** Scenario A offers only marginal gains and/or carries moderate risk. Seek greater FCF impact ($>15\%$ increase) before executing."
        st.warning(rec_a)
    
    st.markdown(f"**Current Action Decision:** {rec_a}")

    # --- Part 2: Contingency Recommendation (Scenario B) ---
    st.subheader("Phase 2: Contingency/Future Strategy Protocol (Scenario B)")
    
    trigger = f"**Switch Condition:** When your current units sold successfully reach **{result_a['Units_Sold']:,.0f}** or higher,"
    
    if result_b['Quick_Ratio'] < QUICK_RATIO_CRITICAL:
        rec_b = "⛔ **PERMANENT AVOIDANCE:** Scenario B is fiscally irresponsible (poor liquidity). Do not use this as a contingency strategy."
        st.error(rec_b)
    elif result_b['Profit'] > result_a['Profit'] * 1.1 and result_b['Loss_Prob'] < result_a['Loss_Prob']: # 10% profit increase AND lower risk
        rec_b = f"🌟 **AUTOMATIC SWITCH RECOMMENDATION:** {trigger} switch to Scenario B immediately. It offers a $10\%+$ higher profit with **LOWER RISK** than Strategy A. This is the optimal path to scale."
        st.success(rec_b)
    elif result_b['Profit'] > result_a['Profit'] and result_b['VaR_95'] > result_a['VaR_95']:
        rec_b = f"📈 **SCALABILITY SWITCH CONSIDERED:** {trigger} you may switch to Scenario B. It yields higher profit ($${result_b['Profit']:,.0f}$$), but accept the risk profile (VaR 95% is higher at $${result_b['VaR_95']:,.0f}$$)."
        st.info(rec_b)
    else:
        rec_b = f"⚠️ **DO NOT SWITCH:** {trigger} continue with Scenario A, as Scenario B does not offer sufficient profit gain or carries excessive new risk."
        st.warning(rec_b)

    st.markdown(f"**Contingency Action Decision:** {rec_b}")


    # ================================================================================
    # MONTE CARLO VISUALIZATION
    # ================================================================================
    st.markdown("---")
    st.subheader("📈 Comparative Profit Distribution & Risk Profile")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scenario A Plot
    ax.hist(result_a['Sim_Profits'], bins=50, color='blue', alpha=0.5, label='A: Immediate Strategy')
    ax.axvline(result_a['Profit'], color='darkblue', linestyle="--")
    ax.axvline(result_a['VaR_95'], color='red', linestyle="-", linewidth=2)

    # Scenario B Plot
    ax.hist(result_b['Sim_Profits'], bins=50, color='orange', alpha=0.5, label='B: Contingency Strategy')
    ax.axvline(result_b['Profit'], color='darkorange', linestyle=":")
    ax.axvline(result_b['VaR_95'], color='red', linestyle="--", linewidth=2)

    ax.axvline(BASE_FINANCE['Avg_Profit'], color='gray', linestyle="-.", label="Base Case Profit")
    
    ax.legend(title="Risk Metrics & Scenarios")
    ax.set_title("Comparative Profit Distribution (5000 Simulations per Scenario)")
    ax.set_xlabel("Predicted Profit ($)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


else:
    # Initial load message
    st.info("👈 **Define your 'Immediate' (A) and 'Future' (B) strategies using the levers on the sidebar. Then, click 'Run AI Strategic Simulation 🤖' to generate a full financial analysis and two-stage decision protocol.**")
