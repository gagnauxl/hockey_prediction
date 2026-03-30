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
    df = dp.load_data()
    dp.define_team_ids(df)     # Initialisiert die Team-IDs basierend auf den Daten, damit sie in der Team-Klasse verfügbar sind
    df = dp.prepare_data(df)  # Bereinigt die Daten, z.B. durch Entfernen unnötiger Spalten, nur noch Features
    df = dp.add_team_ids(df)   # Konvertiert die Teamnamen in numerische IDs, damit sie für das Modell verwendet werden können
    df = dp.add_target_variables(df) # Fügt die Zielvariable hinzu, z.B. durch Berechnung des Spielausgangs basierend auf den Ergebnissen
    X, y = dp.Select_Features_Target(df)  # Wählt die relevanten Features (Home, Away) und die Zielvariable (Target) aus, um sie für das Modelltraining vorzubereiten
    print(f"\nShape of dataframe X (rows, columns): {X.shape}")
    X.info()
    print(X.head())
    print(f"\nShape of target variable y: {y.shape}")
    print(f"Last 10 values of df: \n{df[['Home', 'Away', 'Resultat', 'OT/SO']].tail(10)}")

    lr.analyze_model_performance(X,y)
    print("end of main")
