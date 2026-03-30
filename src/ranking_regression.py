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
    
    if points_a > points_b:
        return 3  # Team A gewinnt regulär
    elif points_a < points_b:
        return 0  # Team B gewinnt regulär
    else:
        return 3  # Home team wins
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

if __name__ == "__main__":
    df = dp.load_data()
    dp.define_team_ids(df)     # Initialisiert die Team-IDs basierend auf den Daten, damit sie in der Team-Klasse verfügbar sind
     
    df = dp.prepare_data(df)  # Bereinigt die Daten, z.B. durch Entfernen unnötiger Spalten, nur noch Features
    df = dp.add_team_ids(df)   # Konvertiert die Teamnamen in numerische IDs, damit sie für das Modell verwendet werden können

    df = dp.add_target_variables(df) # Fügt die Zielvariable hinzu, z.B. durch Berechnung des Spielausgangs basierend auf den Ergebnissen

    X, y = dp.Select_Features_Target(df)  # Wählt die relevanten Features (Home, Away) und die Zielvariable (Target) aus, um sie für das Modelltraining vorzubereiten

    ranking_df = dp.create_team_ranking(df, 5)  # Erstellt die Rangliste der Teams basierend auf den Punkten nach 5 Runden (Hin und Zurück)
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
