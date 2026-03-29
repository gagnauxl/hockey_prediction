#!/usr/bin/python
# to handle datasets
import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from team import Team

#------- Data processing functions -------
def load_data() -> pd.DataFrame:
    # https://www.sihf.ch/de/game-center/national-league/#/standing/rank/asc/page/0/
    datafile = os.path.join(os.path.dirname(__file__), "../data/regular-season-2526.csv")
    df = pd.read_csv(datafile, sep=';')
    # print("Shape of dataframe df (rows, columns): {}".format(df.shape))
    # df.info()
    # print(df.head())
    return df

# call it once to initialize the team mappings based on the data, so that they are available in the Team class
# Usage: Team.Id(str) -> int, Team.Name(int) -> str
def define_team_ids(df: pd.DataFrame) -> list[str]:
    teams = pd.concat([df['Home'], df['Away']]).unique()
    Team.initialize(teams.tolist())

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Entfernen von unnötigen Spalten, z.B. 'Date', 'Time', 'Attendance', 'Venue', 'City', 'State'
    df_cleaned = df[['Home', 'Away', 'Resultat', 'OT/SO', 'Zus.']]
    #df_cleaned = df.drop(columns=['Tag', 'Datum', 'Zeit', 'Drittel', 'Status', 'Versch.', 'Stadion', 'TV/Online', 'Id', 'Liga', 'Region', 'Phase'], errors='ignore')
    #df_cleaned.info()
    #print(df_cleaned.head())
    return df_cleaned

def add_team_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Home_Id'] = df['Home'].map(Team.Id)
    df['Away_Id'] = df['Away'].map(Team.Id)
    print(df.head())
    return df

def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
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
    def determine__home_goals(result) -> int:
        home_goals, away_goals = map(int, result.split(':'))
        return home_goals
    
    def determine__away_goals(result) -> int:
        home_goals, away_goals = map(int, result.split(':'))
        return away_goals
        
    df = df.copy()
    df['Points'] = df[['OT/SO', 'Resultat']].apply(lambda row: determine_winner(row['OT/SO'], row['Resultat']), axis=1)
    df['Home_Goals'] = df[['Resultat']].apply(lambda row: determine__home_goals(row['Resultat']), axis=1)
    df['Away_Goals'] = df[['Resultat']].apply(lambda row: determine__away_goals(row['Resultat']), axis=1)
    print(df.head(10))
    return df

def add_round(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Round'] = (df.index // 7) + 1
    return df   


"""
Rangliste der Teams erstellen, basierend auf den Ergebnissen der Spiele, z.B. durch Berechnung der Gesamtpunkte für jedes Team (3 Punkte für einen regulären Sieg, 2 Punkte für einen Sieg in Overtime/Shootout, 1 Punkt für eine Niederlage in Overtime/Shootout, 0 Punkte für eine reguläre Niederlage) und Sortieren der Teams nach Punkten. Diese Rangliste könnte dann als Grundlage für die Vorhersage von Spielergebnissen dienen, indem
In:
 - Spielergebnisse: pd.DataFrame mit folgenden Spalten:
   Tag       Datum   Zeit                       Home         Away Resultat           Drittel  ...                                           Stadion  Zus.            TV/Online              Id  Liga Region           Phase   
0  Di  09.09.2025  19:45            HC Ambri-Piotta   EHC Kloten      2:1       0:0|1:1|1:0  ...                     Gottardo Arena, 6775 Ambri TI  6149             MYSPORTS  20261105000001    NL     CH  Regular Season   
1  Di  09.09.2025  19:45                   HC Davos  Lausanne HC      4:1       0:0|1:0|3:1  ...            zondacrypto-Arena, 7270 Davos-Platz GR  3696             MYSPORTS  20261105000002    NL     CH  Regular Season   
2  Di  09.09.2025  19:45          Fribourg-Gottéron    HC Lugano      3:2   1:0|0:1|1:1|1:0  ...                       BCF Arena, 1700 Fribourg FR  9280  MYSPORTS,TELETICINO  20261105000003    NL     CH  Regular Season   
3  Di  09.09.2025  19:45         Genève-Servette HC     HC Ajoie      5:3       0:1|2:0|3:2  ...                  Les Vernets, 1227 Les Acacias GE  6003             MYSPORTS  20261105000004    NL     CH  Regular Season   
4  Di  09.09.2025  19:45  SC Rapperswil-Jona Lakers   SCL Tigers      5:1       2:0|2:0|1:1  ...  St.Galler Kantonalbank Arena, 8640 Rapperswil SG  4293             MYSPORTS  20261105000005    NL     CH  Regular Season   

 - Datum: str

Out:
    - Rangliste der Teams: pd.DataFrame mit Spalten 'Team', 'Punkte', sortiert nach Punkten in absteigender Reihenfolge
"""
def create_team_ranking(df: pd.DataFrame, rounds: int) -> pd.DataFrame:
    # Berechnung der Punkte für jedes Team basierend auf den Spielergebnissen der ersten 'rounds' Runden
    team_points = {}
    team_games = {}
    games_per_round = len(df['Home'].unique())
    max_games = games_per_round * rounds
    
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= max_games:
            break
            
        home_team = row['Home']
        away_team = row['Away']
        points = row['Points']
        
        if home_team not in team_points:
            team_points[home_team] = 0
            team_games[home_team] = 0
        if away_team not in team_points:
            team_points[away_team] = 0
            team_games[away_team] = 0
        
        team_games[home_team] += 1
        team_games[away_team] += 1
        
        if points == 3:  # regulärer Sieg für das Heimteam
            team_points[home_team] += 3
        elif points == 2:  # Sieg in Overtime/Shootout für das Heimteam
            team_points[home_team] += 2
            team_points[away_team] += 1
        elif points == 1:  # Sieg in Overtime/Shootout für das Auswärtsteam
            team_points[home_team] += 1
            team_points[away_team] += 2
        elif points == 0:  # reguläre Niederlage für das Heimteam
            team_points[away_team] += 3
    
    # Erstellung eines DataFrames aus dem Dictionary und Sortierung nach Punkten
    ranking_df = pd.DataFrame([
        {'Team': team, 'Punkte': team_points[team], 'Spiele': team_games[team]}
        for team in team_points
    ])
    ranking_df = ranking_df.sort_values(by='Punkte', ascending=False).reset_index(drop=True)
    
    return ranking_df

#------- Plotting functions -------
def plot_results(df: pd.DataFrame, model: LinearRegression):
    # Hier könnte man z.B. die tatsächlichen Ergebnisse vs. die vorhergesagten Ergebnisse plotten, um die Leistung des Modells zu visualisieren
    x = df.index
    y = df['Points']
    #Y = model.predict(df[['Home_Id', 'Away_Id']])
    plt.scatter(x, y, label='Data Points')
    y_hut = model.predict(df[['Home_Id', 'Away_Id']]) if model is not None else None
    if y_hut is not None:
        plt.plot(x, y_hut, color='red', label='Fitted Line') 
    plt.xlabel('x') 
    plt.ylabel('y')
    plt.title('Points vs. Index')
    plt.legend()
    plt.show()

"""
Scatter Plot Results
In:
    - y: np.ndarray with actual points
    - y_hut: np.ndarray with predicted points
"""
def plot_actual_vs_predicted(y: np.ndarray, y_hut: np.ndarray, n: int = 50):
    x = np.arange(min(n, len(y)))
    plt.scatter(x, y[:n], color='blue', label='Actual')
    plt.scatter(x, y_hut[:n], color='red', label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Points')
    plt.title('Actual (blue) vs Predicted (red)')
    #plt.legend()
    plt.show()

#------- Evaluation functions -------
def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y - y_pred) ** 2))

def R_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

#------- Modeling functions -------
def Select_Features_Target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[['Home_Id', 'Away_Id']]
    y = df['Points']
    #print(f"Shape of X: {X.shape}, Shape of y: {y.shape}") # 14 * 52 / 2 = 364 samples, 2 features
    return X, y

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
    df = load_data()
    define_team_ids(df)     # Initialisiert die Team-IDs basierend auf den Daten, damit sie in der Team-Klasse verfügbar sind
    #print(f"Team Name to ID mapping: {Team.name_to_id}")
    #print(f"Team ID to Name mapping: {Team.id_to_name}")
    
    df = prepare_data(df)  # Bereinigt die Daten, z.B. durch Entfernen unnötiger Spalten, nur noch Features
    df = add_team_ids(df)   # Konvertiert die Teamnamen in numerische IDs, damit sie für das Modell verwendet werden können

    df = add_target_variables(df) # Fügt die Zielvariable hinzu, z.B. durch Berechnung des Spielausgangs basierend auf den Ergebnissen
    
    ranking_df = create_team_ranking(df, 5)  # Erstellt die Rangliste der Teams basierend auf den Punkten nach 5 
    print("Team Ranking:")
    print(ranking_df)

    # print(f"Predicted outcome for ZSC Lions vs EV Zug: {predict_game_outcome(ranking_df, 'ZSC Lions', 'EV Zug')}")
    # print(f"Predicted outcome for EV Zug vs ZSC Lions: {predict_game_outcome(ranking_df, 'EV Zug', 'ZSC Lions')}")
    # print(f"Predicted outcome for HC Davos vs HC Ajoie: {predict_game_outcome(ranking_df, 'HC Davos', 'HC Ajoie')}")
    # print(f"Predicted outcome for HC Ajoie vs HC Davos: {predict_game_outcome(ranking_df, 'HC Ajoie', 'HC Davos')}")

    X, y = Select_Features_Target(df)  # Wählt die relevanten Features (Home, Away) und die Zielvariable (Target) aus, um sie für das Modelltraining vorzubereiten
    print(f"Custom predicted outcomes for the first 10 games: {predict_game_outcome_custom(ranking_df, X[0:10])}")
    
    model = fit(X, y)          # Trainiert das Modell mit den ausgewählten Features und der Zielvariable
    plot_results(df, model)    # Hier könnte man die tatsächlichen Ergebnisse vs. die vorhergesagten Ergebnisse plotten, um die Leistung des Modells zu visualisieren   
    y_hut = predict(model, X)
    rmse_value = rmse(y, y_hut)
    print(f"Model RMSE: {rmse_value}, y_hut: {y_hut[:10]}")
    print(f"Model R^2: {R_squared(y.to_numpy(), y_hut)}")


    y_hut_approx = predict_rounded(y_hut)
    rmse_value = rmse(y, y_hut_approx)
    print(f"Model RMSE Rounded: {rmse_value}, y_hut_approx: {y_hut_approx[:10]}")
    print(f"Model R^2 Rounded: {R_squared(y.to_numpy(), y_hut_approx)}")
    accuracy = accuracy_score(y.astype(int), y_hut_approx.astype(int))
    print(f"Accuracy: {accuracy}")
    
    y_hut_1_7 = np.full_like(y_hut, 1.7, dtype=float)
    rmse_value = rmse(y, y_hut_1_7)
    print(f"Model RMSE Approximated 1.7: {rmse_value}, y_hut_1_7: {y_hut_1_7[:10]}")

    y_hut_based_on_ranking = predict_game_outcome_custom(ranking_df, X)
    rmse_value = rmse(y, y_hut_based_on_ranking)
    print(f"Model RMSE based on ranking: {rmse_value}, y_hut_based_on_ranking: {y_hut_based_on_ranking[:10]}")
    accuracy = accuracy_score(y.astype(int), y_hut_based_on_ranking.astype(int))
    print(f"Accuracy: {accuracy}")

    y_hut_based_on_ranking_rounded = predict_rounded(y_hut_based_on_ranking)
    rmse_value = rmse(y, y_hut_based_on_ranking_rounded)
    print(f"Model RMSE based on ranking Rounded: {rmse_value}, y_hut_based_on_ranking_rounded: {y_hut_based_on_ranking_rounded[:10]}")
    accuracy = accuracy_score(y.astype(int), y_hut_based_on_ranking.astype(int))
    print(f"Accuracy: {accuracy}")

    y_test = np.array([3,3,2,3,3,3,3,0,0,2])
    y_test_hut = np.array([3,3,3,3,3,3,3,0,0,0])
    rmse_value = rmse(y_test, y_test_hut)
    print(f"Model MSE for custom y and y_hut: {rmse_value}, y_test: {y_test}, y_test_hut: {y_test_hut}")

    y_test = np.array([3,3,3,3,3,3,3,3,3,3])
    y_test_hut = np.array([0,0,0,0,0,0,0,0,0,0])
    rmse_value = rmse(y_test, y_test_hut)
    print(f"Model MSE worst case y and y_hut: {rmse_value}, y_test: {y_test}, y_test_hut: {y_test_hut}")

    print(f"model prediction for the first 10 samples: {model.predict(X[0:10])}")
    print(f"ZSC Lions - EV Zug: {model.predict(pd.DataFrame([[Team.Id('ZSC Lions'), Team.Id('EV Zug')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"EV Zug - ZSC Lions: {model.predict(pd.DataFrame([[Team.Id('EV Zug'), Team.Id('ZSC Lions')]], columns=['Home_Id', 'Away_Id']))}")

    print(f"HC Davos - HC Ajoie: {model.predict(pd.DataFrame([[Team.Id('HC Davos'), Team.Id('HC Ajoie')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"HC Ajoie - HC Davos: {model.predict(pd.DataFrame([[Team.Id('HC Ajoie'), Team.Id('HC Davos')]], columns=['Home_Id', 'Away_Id']))}")

    from sklearn.neighbors import KNeighborsClassifier
    model_knn  = KNeighborsClassifier(n_neighbors=1)
    model_knn.fit(X, y)
    print(f"KNN model prediction for the first 10 samples: {model_knn.predict(X[0:10])}")
    print(f"KNN ZSC Lions - EV Zug: {model_knn.predict(pd.DataFrame([[Team.Id('ZSC Lions'), Team.Id('EV Zug')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"KNN EV Zug - ZSC Lions: {model_knn.predict(pd.DataFrame([[Team.Id('EV Zug'), Team.Id('ZSC Lions')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"KNN HC Davos - HC Ajoie: {model_knn.predict (pd.DataFrame([[Team.Id('HC Davos'), Team.Id('HC Ajoie')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"KNN HC Ajoie - HC Davos: {model_knn.predict(pd.DataFrame([[Team.Id('HC Ajoie'), Team.Id('HC Davos')]], columns=['Home_Id', 'Away_Id']))}")

    y_model = model.predict(X)
    print(f"Linear Regression R^2: {R_squared(y.to_numpy(), y_model)}")

    y_model_knn = model_knn.predict(X)
    accuracy = accuracy_score(y.astype(int), y_model_knn.astype(int))
    print(f"KNN Accuracy: {accuracy}")