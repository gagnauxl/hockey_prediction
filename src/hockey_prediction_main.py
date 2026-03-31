#!/usr/bin/python
# to handle datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# own modules
import data_service as dp
from team import Team
import ranking_regression as rr
import lin_regression as lr
import knn_regression as knn

#------- Modeling functions -------
def fit(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model: LinearRegression, X: pd.DataFrame) -> np.ndarray:
    y = model.predict(X)
    return y

def predict_rounded(y: np.ndarray) -> np.ndarray:
    mean = np.mean(y)
    print(f"Mean of predicted values: {mean}")
    y_ret = np.where(y < mean, 0, 3)
    return y_ret

def evaluate_model(model: LinearRegression, X: pd.DataFrame, y: pd.Series) -> float:
    score = model.score(X, y)
    return score

if __name__ == "__main__":
    """Todo:
    - knn-fit
    - signifikante Features identifizieren, z.B. Zuschauerzahl
    """

    X, y, df = dp.load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf   

    print(f"\nShape of dataframe X (rows, columns): {X.shape}")
    X.info()
    print(X.head())
    print(f"\nShape of target variable y: {y.shape}")
    print(f"Last 10 values of df: \n{df[['Home', 'Away', 'Resultat', 'OT/SO']].tail(10)}")

    lr.analyze_model_performance(X,y)
    #rr.analyze_model_performance(df,y)
    #knn.analyze_model_performance(X,y)
    print("end of main")
