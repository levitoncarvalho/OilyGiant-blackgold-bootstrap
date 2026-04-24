import pandas as pd
from src.config import TOTAL_WELLS_CHOSEN, REVENUE_PER_UNIT, TOTAL_BUDGET_USD

def calculate_profit(target, predictions, count=TOTAL_WELLS_CHOSEN):
    # Select the top count wells based on predicted volume, then compute real profit using the actual reserves (target)
    pred_sorted = predictions.sort_values(ascending=False)
    selected_targets = target[pred_sorted.index][:count]
    total_volume = selected_targets.sum()
    revenue = total_volume * REVENUE_PER_UNIT
    profit = revenue - TOTAL_BUDGET_USD

    return profit