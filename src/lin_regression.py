#!/usr/bin/python
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# own modules
from team import Team
import data_service as dp

#------- Modeling functions -------
def fit(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_rounded(y: float) -> float:
    if y < 1.5: 
        return 0.0
    return 3.0

if __name__ == "__main__":
    df = dp.load_data()
    dp.define_team_ids(df)     # Initialisiert die Team-IDs basierend auf den Daten, damit sie in der Team-Klasse verfügbar sind
     
    df = dp.prepare_data(df)  # Bereinigt die Daten, z.B. durch Entfernen unnötiger Spalten, nur noch Features
    df = dp.add_team_ids(df)   # Konvertiert die Teamnamen in numerische IDs, damit sie für das Modell verwendet werden können

    df = dp.add_target_variables(df) # Fügt die Zielvariable hinzu, z.B. durch Berechnung des Spielausgangs basierend auf den Ergebnissen

    X, y = dp.Select_Features_Target(df)  # Wählt die relevanten Features (Home, Away) und die Zielvariable (Target) aus, um sie für das Modelltraining vorzubereiten
   
    model = fit(X, y)             # Trainiert das Modell mit den ausgewählten Features und der Zielvariable
    #dp.plot_results(df, model)

    y = y[0:10].to_numpy()
    y_hut = model.predict(X[0:10])
    y_hut_rounded = np.array([predict_rounded(val) for val in y_hut])
    print(f"Actual outcomes for the first 10 games: {y}")
    print(f"                             Predicted: {y_hut}")
    print(f"                     Predicted rounded: {y_hut_rounded}")

    print(f"RMSE, y vs y_hut: {dp.rmse(y, y_hut)}")
    print(f"RMSE Rounded, y vs y_hut_rounded: {dp.rmse(y, y_hut_rounded)}")
    accuracy = accuracy_score(y.astype(int), y_hut.astype(int))
    print(f"Accuracy, y vs y_hut: {accuracy}")
    accuracy = accuracy_score(y.astype(int), y_hut_rounded.astype(int))
    print(f"Accuracy, y vs y_hut_rounded: {accuracy}")

    print("Specific game predictions:")
    print(f"ZSC Lions - EV Zug: {model.predict(pd.DataFrame([[Team.Id('ZSC Lions'), Team.Id('EV Zug')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"EV Zug - ZSC Lions: {model.predict(pd.DataFrame([[Team.Id('EV Zug'), Team.Id('ZSC Lions')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"HC Davos - HC Ajoie: {model.predict(pd.DataFrame([[Team.Id('HC Davos'), Team.Id('HC Ajoie')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"HC Ajoie - HC Davos: {model.predict(pd.DataFrame([[Team.Id('HC Ajoie'), Team.Id('HC Davos')]], columns=['Home_Id', 'Away_Id']))}")

    print("Specific game predictions rounded:")
    print(f"ZSC Lions - EV Zug: {predict_rounded(
        model.predict(pd.DataFrame([[Team.Id('ZSC Lions'), Team.Id('EV Zug')]], columns=['Home_Id', 'Away_Id'])))}")
    print(f"EV Zug - ZSC Lions: {predict_rounded(
        model.predict(pd.DataFrame([[Team.Id('EV Zug'), Team.Id('ZSC Lions')]], columns=['Home_Id', 'Away_Id'])))}")
    print(f"HC Davos - HC Ajoie: {predict_rounded(
        model.predict(pd.DataFrame([[Team.Id('HC Davos'), Team.Id('HC Ajoie')]], columns=['Home_Id', 'Away_Id'])))}")
    print(f"HC Ajoie - HC Davos: {predict_rounded(
        model.predict(pd.DataFrame([[Team.Id('HC Ajoie'), Team.Id('HC Davos')]], columns=['Home_Id', 'Away_Id'])))}")
