#!/usr/bin/python
# to handle datasets
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from team import Team

def load_data() -> pd.DataFrame:
    # https://www.sihf.ch/de/game-center/national-league/#/standing/rank/asc/page/0/
    datafile = os.path.join(os.path.dirname(__file__), "../data/regular-season-2526.csv")
    df = pd.read_csv(datafile, sep=';')
    print("Shape of dataframe df (rows, columns): {}".format(df.shape))
    df.info()
    print(df.head())
    return df

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Entfernen von unnötigen Spalten, z.B. 'Date', 'Time', 'Attendance', 'Venue', 'City', 'State'
    df_cleaned = df[['Home', 'Away', 'Resultat', 'OT/SO', 'Zus.']]
    #df_cleaned = df.drop(columns=['Tag', 'Datum', 'Zeit', 'Drittel', 'Status', 'Versch.', 'Stadion', 'TV/Online', 'Id', 'Liga', 'Region', 'Phase'], errors='ignore')
    df_cleaned.info()
    print(df_cleaned.head())
    return df_cleaned

def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    # Zielvariable erstellen: 1 für Heimsieg, 0 für Auswärtssieg, 0.5 für Unentschieden
    def determine_winner(type, result):
        home_goals, away_goals = map(int, result.split(':'))
        if type == 'OT' or type == 'SO':
            if home_goals > away_goals:
                return 2
            elif home_goals < away_goals:
                return 1
            else:
                return np.nan  # Für ungültige Werte
        else:
            if home_goals > away_goals:
                return 3
            elif home_goals < away_goals:
                return 0
            else:
                return np.nan  # Für ungültige Werte

    df = df.copy()
    df['Target'] = df[['OT/SO', 'Resultat']].apply(lambda row: determine_winner(row['OT/SO'], row['Resultat']), axis=1)
    print(df.head(10))
    return df

def Select_Features_Target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[['Home', 'Away']]
    y = df['Target']
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

#  shift to a class for better structure and to avoid global variables
def define_team_ids(df: pd.DataFrame) -> list[str]:
    teams = pd.concat([df['Home'], df['Away']]).unique()
    Team.initialize(teams.tolist())

def evaluate_model(model: LinearRegression, X: pd.DataFrame, y: pd.Series) -> float:
    score = model.score(X, y)
    print(f"Model R^2 score: {score}")
    return score

def convert_team_names_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X_numeric = X.copy()
    X_numeric['Home'] = X_numeric['Home'].map(Team.Id)
    X_numeric['Away'] = X_numeric['Away'].map(Team.Id)
    print(X_numeric.head())
    return X_numeric

def fit(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    df = load_data()
    define_team_ids(df)     # Initialisiert die Team-IDs basierend auf den Daten, damit sie in der Team-Klasse verfügbar sind
    print(f"Team Name to ID mapping: {Team.name_to_id}")
    print(f"Team ID to Name mapping: {Team.id_to_name}")
    
    df_cleaned = prepare_data(df)  # Bereinigt die Daten, z.B. durch Entfernen unnötiger Spalten, nur noch Features

    df_with_target = add_target_variable(df_cleaned) # Fügt die Zielvariable hinzu, z.B. durch Berechnung des Spielausgangs basierend auf den Ergebnissen
    
    X, y = Select_Features_Target(df_with_target)  # Wählt die relevanten Features (Home, Away) und die Zielvariable (Target) aus, um sie für das Modelltraining vorzubereiten
    X_numeric = convert_team_names_to_numeric(X)   # Konvertiert die Teamnamen in numerische IDs, damit sie für das Modell verwendet werden können
    model = fit(X_numeric, y)

    print(f"model prediction for the first 10 samples: {model.predict(X_numeric[0:10])}")

    print(f"ZSC Lions - EV Zug: {model.predict(pd.DataFrame([[Team.Id('ZSC Lions'), Team.Id('EV Zug')]], columns=['Home', 'Away']))}")
    print(f"EV Zug - ZSC Lions: {model.predict(pd.DataFrame([[Team.Id('EV Zug'), Team.Id('ZSC Lions')]], columns=['Home', 'Away']))}")

    print(f"HC Davos - HC Ajoie: {model.predict(pd.DataFrame([[Team.Id('HC Davos'), Team.Id('HC Ajoie')]], columns=['Home', 'Away']))}")
    print(f"HC Ajoie - HC Davos: {model.predict(pd.DataFrame([[Team.Id('HC Ajoie'), Team.Id('HC Davos')]], columns=['Home', 'Away']))}")