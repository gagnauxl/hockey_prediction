#!/usr/bin/python
# to handle datasets
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

def convert_team_names_to_numeric(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    teams = pd.concat([X['Home'], X['Away']]).unique()
    team_to_id = {team: idx for idx, team in enumerate(teams)}
    X_numeric = X.copy()
    X_numeric['Home'] = X_numeric['Home'].map(team_to_id).astype(int)
    X_numeric['Away'] = X_numeric['Away'].map(team_to_id).astype(int)
    print(f"Team to ID mapping: {team_to_id}")
    print(X_numeric.head())
    return X_numeric, team_to_id

def create_id_to_team_mapping(team_to_id: dict) -> dict:
    return {idx: team for team, idx in team_to_id.items()}

def name_to_id(team_name: str, team_to_id: dict) -> int:
    return team_to_id.get(team_name, -1)  # Rückgabe von -1, wenn der Teamname nicht gefunden wird

def id_to_name(team_id: int, id_to_team: dict) -> str:
    return id_to_team.get(team_id, "Unknown Team")  # Rückgabe von "Unknown Team", wenn die ID nicht gefunden wird

def fit(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    df = load_data()
    df_cleaned = prepare_data(df)
    df_with_target = add_target_variable(df_cleaned)
    X, y = Select_Features_Target(df_with_target)
    X_numeric, team_to_id = convert_team_names_to_numeric(X)
    print(f"team_to_id: {team_to_id}")
    id_to_team = create_id_to_team_mapping(team_to_id)
    print(f"id_to_team: {id_to_team}")  
    print(f"name of team with ID 0: {id_to_name(0, id_to_team)}")
    model = fit(X_numeric, y)

    print(f"model prediction for the first 10 samples: {model.predict(X_numeric[0:10])}")

    print(f"ZSC Lions - EV Zug: {model.predict(pd.DataFrame([[name_to_id('ZSC Lions', team_to_id), name_to_id('EV Zug', team_to_id)]], columns=['Home', 'Away']))}")
    print(f"EV Zug - ZSC Lions: {model.predict(pd.DataFrame([[name_to_id('EV Zug', team_to_id), name_to_id('ZSC Lions', team_to_id)]], columns=['Home', 'Away']))}")

    print(f"HC Davos - HC Ajoie: {model.predict(pd.DataFrame([[name_to_id('HC Davos', team_to_id), name_to_id('HC Ajoie', team_to_id)]], columns=['Home', 'Away']))}")
    print(f"HC Ajoie - HC Davos: {model.predict(pd.DataFrame([[name_to_id('HC Ajoie', team_to_id), name_to_id('HC Davos', team_to_id)]], columns=['Home', 'Away']))}")