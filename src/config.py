from pathlib import Path

# Business and simulation constants
TOTAL_BUDGET_USD = 100_000_000
TOTAL_WELLS_CHOSEN = 200
REVENUE_PER_UNIT = 4_500            # USD per 1,000 barrels
RISK_THRESHOLD = 0.025              # 2.5% maximum acceptable loss risk
RANDOM_STATE = 54321
N_BOOTSTRAP = 1000                  # bootstrap iterations
SAMPLE_SIZE = 500                   # survey points per region (as per project rules)

#paths 
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'