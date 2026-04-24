import pandas as pd
from src.config import DATA_DIR

def load_all_regions():
    # Load the 3 region datasets, drop the non-predictive 'id' column and return a dict
    regions = {}
    for i in range(3):
        path = DATA_DIR / f'geo_data_{i}.csv'
        df = pd.read_csv(path)
        df = df.drop('id', axis=1) 
        regions[f"region_{i}"] = df
    return regions