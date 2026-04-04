#!/usr/bin/python
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# own modules
from team import Team

"""
Data Service Module
- data processing
- feature engineering
- target variable creation
- team ranking creation
- plotting
- statistics: RMSE, R²
"""

#------- Data processing functions -------
def load_data() -> pd.DataFrame:
    # https://www.sihf.ch/de/game-center/national-league/#/standing/rank/asc/page/0/
    datafile = os.path.join(os.path.dirname(__file__), "../data/regular-season-2526.csv")
    df = pd.read_csv(datafile, sep=';')
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
    #print(df.head())
    return df

"""
Fügt die Zielvariable 
   Points
   Home_Goals und Away_Goals 
"""
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
    #print(df.head(10))
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
def create_team_ranking(df: pd.DataFrame, train_idx_start: int, train_idx_end: int) -> pd.DataFrame:
    # Berechnung der Punkte für jedes Team basierend auf den Spielergebnissen der ersten 'rounds' Runden
    team_points = {}
    team_games = {}
    GAMES_PER_ROUND = 7
    for i, (_, row) in enumerate(df.iloc[train_idx_start*GAMES_PER_ROUND:train_idx_end*GAMES_PER_ROUND]
                                 .iterrows(), start=train_idx_start*GAMES_PER_ROUND):
            
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

def Select_Features_Target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[['Home_Id', 'Away_Id']]   # macht eine Kopie
    y = df['Points']
    #print(f"Shape of X: {X.shape}, Shape of y: {y.shape}") # 14 * 52 / 2 = 364 samples, 2 features
    return X, y

"""not used for the moment """
def add_round(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Round'] = (df.index // 7) + 1
    return df

def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = load_data()
    define_team_ids(df)     # Initialisiert die Team-IDs basierend auf den Daten, damit sie in der Team-Klasse verfügbar sind
    df = prepare_data(df)  # Bereinigt die Daten, z.B. durch Entfernen unnötiger Spalten, nur noch Features
    df = add_team_ids(df)   # Konvertiert die Teamnamen in numerische IDs, damit sie für das Modell verwendet werden können
    df = add_target_variables(df) # Fügt die Zielvariable hinzu, z.B. durch Berechnung des Spielausgangs basierend auf den Ergebnissen
    X, y = Select_Features_Target(df)  # Wählt die relevanten Features (Home, Away) und die Zielvariable (Target) aus, um sie für das Modelltraining vorzubereiten
    return X, y, df

#------- Plotting functions -------
def plot_results(df: pd.DataFrame, model: LinearRegression):
    # Hier könnte man z.B. die tatsächlichen Ergebnisse vs. die vorhergesagten Ergebnisse plotten, um die Leistung des Modells zu visualisieren
    x = df.index
    y = df['Points']
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
def plot_actual_vs_predicted(y: np.ndarray, y_hut: np.ndarray):
    x = np.arange(len(y))
    #print(f"y    : {y.to_numpy()}\ny_hut: {y_hut}")
    plt.scatter(x, y, color='blue', label='Actual')
    plt.scatter(x, y_hut, color='red', label='Predicted')
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

def score(y_test: np.ndarray, y_hut: np.ndarray):
    rmse_value = rmse(y_test, y_hut)
    r_squared_value = R_squared(y_test, y_hut)
    accuracy = accuracy_score(y_test.astype(int), y_hut.astype(int))
    print(f"Model RMSE: {rmse_value}")
    print(f"Model R^2: {r_squared_value}")
    print(f"Model Accuracy: {accuracy}")
    plot_actual_vs_predicted(y_test, y_hut)

def score_last_10_games(y_test: np.ndarray, y_hut: np.ndarray):
    y_hut_last_10 = y_hut[-10:]
    y_test_last_10 = y_test[-10:]
    print(f"Actual outcomes for the last 10 games:\n {y_test_last_10}")
    print(f"                             Predicted: {y_hut_last_10}")
 
    print(f"RMSE, y vs y_hut: {rmse(y_test_last_10, y_hut_last_10)}")
    accuracy = accuracy_score(y_test_last_10.astype(int), y_hut_last_10.astype(int))
    print(f"Accuracy last 10 games, y vs y_hut: {accuracy}")

    plot_actual_vs_predicted(y_test_last_10, y_hut_last_10)

if __name__ == "__main__":
    X, y, df = load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf
    print(f"Team Name to ID mapping:\n{Team.name_to_id}")
    print(f"\nTeam ID to Name mapping:\n{Team.id_to_name}")
    print(f"\nShape of dataframe df (rows, columns): {df.shape}")
    df.info()
    print(df.head())
    ranking_df = create_team_ranking(df, 0, 52)
    print(f"Team Ranking:\n{ranking_df}")
    print(f"Team Ranking letzte Runde:\n{create_team_ranking(df, 51, 52)}")
    print(f"Last 7 games:\n{df[['Home_Id', 'Away_Id', 'Home', 'Away', 'Resultat', 'OT/SO', 'Points']].tail(7).to_string()}")


    """
    Data columns (total 10 columns):
    #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
    0   Home        364 non-null    object
    1   Away        364 non-null    object
    2   Resultat    364 non-null    object
    3   OT/SO       83 non-null     object
    4   Zus.        364 non-null    int64
    5   Home_Id     364 non-null    int64
    6   Away_Id     364 non-null    int64
    7   Points      364 non-null    int64
    8   Home_Goals  364 non-null    int64
    9   Away_Goals  364 non-null    int64
    dtypes: int64(6), object(4)
    memory usage: 28.6+ KB
                            Home         Away Resultat OT/SO  Zus.  Home_Id  Away_Id  Points  Home_Goals  Away_Goals
    0            HC Ambri-Piotta   EHC Kloten      2:1   NaN  6149        0       10       3           2           1
    1                   HC Davos  Lausanne HC      4:1   NaN  3696        1       12       3           4           1
    2          Fribourg-Gottéron    HC Lugano      3:2    OT  9280        2       11       2           3           2
    3         Genève-Servette HC     HC Ajoie      5:3   NaN  6003        3        8       3           5           3
    4  SC Rapperswil-Jona Lakers   SCL Tigers      5:1   NaN  4293        4       13       3           5           1
    """

