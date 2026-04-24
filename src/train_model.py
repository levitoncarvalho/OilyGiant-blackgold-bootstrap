import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.config import RANDOM_STATE

def train_and_evaluate(df):
    #Split the data, fit a linear regression and return the trained model, the validation target, the predictions on validation set and the RMSE
    X = df.drop('product', axis=1)
    y = df['product']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    predictions = pd.Series(predictions, index=y_valid.index)

    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    return model, y_valid, predictions, rmse