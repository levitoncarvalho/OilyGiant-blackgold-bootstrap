from src.data_loader import load_all_regions
from src.train_model import train_and_evaluate
from src.bootstrap import evaluate_risk

def main():
    regions = load_all_regions()
    results = {}

    for name, df in regions.items():
        i, y_valid, preds, rmse = train_and_evaluate(df)
        metrics = evaluate_risk(y_valid, preds)
        results[name] = {
            'rmse': rmse,
            'metrics': metrics
        }

        print(f"{name}:")
        print(f"RMSE: {rmse:.2f} thousand barrels")
        print(f"Mean Profit: ${metrics['mean_profit']:,.2f}")
        print(f"Risk of Loss: {metrics['risk_of_loss']:.1%}")
        print(f"Approved {metrics['is_approved']}\n")

    #Select the best approved region
    approved = {k: v for k, v in results.items() if v['metrics']['is_approved']}
    if approved:
        best_region = max(approved, key=lambda k:approved[k]['metrics']['mean_profit'])
        print(f"Recommended region: {best_region}")
    else:
        print(f"No region satisfies the risk criterion")

if __name__ == "__main__":
    main()