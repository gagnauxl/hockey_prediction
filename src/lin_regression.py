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
def separate_train_test(df: pd.DataFrame, test_size: float = 0.2) -> (pd.DataFrame, pd.DataFrame):
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def analyze_model_performance(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, 
                              train_idx_start: int, train_idx_end: int,
                              test_idx_start: int, test_idx_end: int):
    # df with features, y with target variable, idxs in rounds
    GAMES_PER_ROUND = 7
    train_df = X.iloc[train_idx_start*GAMES_PER_ROUND:train_idx_end*GAMES_PER_ROUND]  # ende wird nicht genommen, also 0-279 für die ersten 40 Runden
    test_df = X.iloc[test_idx_start*GAMES_PER_ROUND:test_idx_end*GAMES_PER_ROUND]
    print(f"Number of training items: {len(train_df)}, test items: {len(test_df)}")
    print(f"Training on rounds {train_idx_start} to {train_idx_end}, testing on rounds {test_idx_start} to {test_idx_end}")
    y_train = y.iloc[train_idx_start*GAMES_PER_ROUND:train_idx_end*GAMES_PER_ROUND]
    y_test = y.iloc[test_idx_start*GAMES_PER_ROUND:test_idx_end*GAMES_PER_ROUND]

    model = fit(train_df, y_train)
    dp.plot_results(df[test_idx_start*GAMES_PER_ROUND:test_idx_end*GAMES_PER_ROUND], model)
    y_hut = model.predict(test_df)
    print(f"Predicted values y_hut) are rounded:")
    y_hut = np.array([predict_rounded(val) for val in y_hut])

    dp.score(y_test, y_hut)
    # dp.score_last_10_games(y_test, y_hut) only for testing

def fit(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_rounded(y: float) -> float:
    if y < 1.5: 
        return 0.0
    return 3.0

if __name__ == "__main__":
    X, y, df = dp.load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf   
    model = fit(X, y)            # Trainiert auf ALLE Daten
    dp.plot_results(df, model)

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
