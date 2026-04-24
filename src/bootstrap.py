from asttokens.util import replace
import numpy as np
import pandas as pd
from src.config import SAMPLE_SIZE, N_BOOTSTRAP, RISK_THRESHOLD, RANDOM_STATE
from src.profit import calculate_profit

def evaluate_risk(target, predictions, random_state=RANDOM_STATE):
    # Perform bootstrap to estimate profit distribution and compute the expected average profit, 95% confidence interval, risk of loss and aproval status. Also returns the full profit distribution for visualization
    rng = np.random.RandomState(random_state)
    profits = []

    for i in range(N_BOOTSTRAP):
        #simluate a survey of SAMPLE_SIZE points with replacement
        target_sample = target.sample(n=SAMPLE_SIZE, replace=True, random_state=rng)
        pred_sample = predictions[target_sample.index]
        profit = calculate_profit(target_sample, pred_sample)
        profits.append(profit)

    profits = pd.Series(profits)
    mean_profit = profits.mean()
    lower_quantile = profits.quantile(0.025)
    upper_quantile = profits.quantile(0.0975)
    risk_of_loss = (profits < 0).mean()
    is_approved = risk_of_loss < RISK_THRESHOLD

    return {
        'mean_profit': mean_profit,
        'lower_quantile': lower_quantile,
        'upper_quantile': upper_quantile,
        'risk_of_loss': risk_of_loss,
        'is_approved': is_approved,
        'distribution': profits
    }