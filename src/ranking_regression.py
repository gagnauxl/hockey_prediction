#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# own modules
from team import Team
import data_service as dp

#------- Modeling functions -------
"""
Predict game based on ranking of teams, e.g. predict that the team with more points in the ranking will win
In: 
    - Rangliste der Teams: pd.DataFrame mit Spalten 'Team', 'Punkte', sortiert nach Punkten in absteigender Reihenfolge
    - Team A, Team B: str
Out:
    - Vorhersage des Spielergebnisses: int (0 für reguläre Niederlage, 3 für Sieg in Overtime/Shootout, 2 für Sieg in Overtime/Shootout, 3 für regulären Sieg)
"""
def predict_game_outcome(ranking_df: pd.DataFrame, team_a: str, team_b: str) -> int:
    points_a = ranking_df.loc[ranking_df['Team'] == team_a, 'Punkte'].values[0]
    points_b = ranking_df.loc[ranking_df['Team'] == team_b, 'Punkte'].values[0]

    threshold = 0
    
    if points_a > points_b + threshold:
        return 3  # Team A gewinnt regulär, weil es deutlich besser ist
    elif points_a > points_b:
        return 2  # Team A gewinnt in Overtime/Shootout, weil es besser ist, aber nicht deutlich
    elif points_a < points_b - threshold:
        return 0  # Team B gewinnt regulär, weil es deutlich besser ist
    elif points_a < points_b:
        return 1  # Team B gewinnt in Overtime/Shootout, weil es besser ist, aber nicht deutlich
    else:
        return 2  # Home team wins Overtime/Shootout, wenn beide Teams gleich stark sind
"""
Predict y_hat based on the ranking of teams
In: 
    - ranking_df: pd.DataFrame mit Spalten 'Team', 'Punkte', sortiert nach Punkten in absteigender Reihenfolge
    - X: pd.DataFrame with columns 'Home_Id', 'Away_Id'
Out:
    - y_hat: np.ndarray with predicted points based on the ranking of teams 

"""
def predict_game_outcome_custom(ranking_df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    y_hat = []
    for _, row in X.iterrows():
        home_team_id = row['Home_Id']
        away_team_id = row['Away_Id']
        
        home_team_name = Team.Name(home_team_id)
        away_team_name = Team.Name(away_team_id)
        outcome = predict_game_outcome(ranking_df, home_team_name, away_team_name)
        y_hat.append(outcome)
    return np.array(y_hat)

def analyze_model_performance(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, 
                              train_idx_start: int, train_idx_end: int,
                              test_idx_start: int, test_idx_end: int):
    # df with features, y with target variable, idxs in rounds
    GAMES_PER_ROUND = 7
    test_df = df.iloc[test_idx_start*GAMES_PER_ROUND:test_idx_end*GAMES_PER_ROUND]
    y_test = y[test_idx_start*GAMES_PER_ROUND:test_idx_end*GAMES_PER_ROUND]
    print(f"Number of training items: {(train_idx_end- train_idx_start)*GAMES_PER_ROUND}, test items: {len(test_df)}")
    print(f"Training on rounds {train_idx_start} to {train_idx_end}, testing on rounds {test_idx_start} to {test_idx_end}")

    ranking_df = dp.create_team_ranking(df, train_idx_start, train_idx_end)
    print(ranking_df)
    y_hut = predict_game_outcome_custom(ranking_df, test_df)

    dp.score(y_test, y_hut)
    #dp.score_last_10_games(y_test, y_hut) #only for test

if __name__ == "__main__":
    X, y, df = dp.load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf   

    ranking_df = dp.create_team_ranking(df, 0, 52)  # Erstellt die Rangliste der Teams basierend auf den Punkten nach 5 Runden (Hin und Zurück)
    print("Team Ranking:")
    print(ranking_df)

    y = y[0:10].to_numpy()
    y_hut = predict_game_outcome_custom(ranking_df, X[0:10])
    print(f"first 10 games:\n{df[0:10]}")
    print(f"Actual outcomes for the first 10 games: {y}")
    print(f"                             Predicted: {y_hut}")

    print(f"RMSE, y vs y_hut: {dp.rmse(y, y_hut)}")
    accuracy = accuracy_score(y.astype(int), y_hut.astype(int))
    print(f"Accuracy, y vs y_hut: {accuracy}")

    print("Specific game predictions:")
    print(f"Predicted outcome for ZSC Lions vs EV Zug: {predict_game_outcome(ranking_df, 'ZSC Lions', 'EV Zug')}")
    print(f"Predicted outcome for EV Zug vs ZSC Lions: {predict_game_outcome(ranking_df, 'EV Zug', 'ZSC Lions')}")
    print(f"Predicted outcome for HC Davos vs HC Ajoie: {predict_game_outcome(ranking_df, 'HC Davos', 'HC Ajoie')}")
    print(f"Predicted outcome for HC Ajoie vs HC Davos: {predict_game_outcome(ranking_df, 'HC Ajoie', 'HC Davos')}")
