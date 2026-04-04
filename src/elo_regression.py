#!/usr/bin/python
from typing_extensions import Final

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import product

# own modules
from team import Team
import data_service as dp

import pandas as pd
import numpy as np


def expected_score(elo_home, elo_away):
    """Erwartete Siegwahrscheinlichkeit des Heimteams."""
    return 1 / (1 + 10 ** ((elo_away - elo_home) / 400))

def update_elo(elo_home, elo_away, score_home, score_away, K):
    """Elo-Update nach einem Spiel."""
    exp_home = expected_score(elo_home, elo_away)
    new_home = elo_home + K * (score_home - exp_home)
    new_away = elo_away + K * (score_away - (1 - exp_home))
    return new_home, new_away

def get_match_score(points, OT_WEIGHT):
    """Konvertiert Punkte (0/1/3) in Elo-Scores."""
    if points == 3:
        return 1, 0
    elif points == 1:
        return OT_WEIGHT, 1 - OT_WEIGHT
    else:
        return 0, 1

# ---------------------------------------------------------
# Elo-Modell anwenden
# ---------------------------------------------------------
#        Home         Away           Resultat OT/SO  Zus.  Home_Id  Away_Id  Points  Home_Goals  Away_Goals
# 0  HC Ambri-Piotta   EHC Kloten      2:1     NaN  6149        0       10       3           2           1
def compute_elo(df, K, HOME_ADV, OT_WEIGHT):
    teams = pd.concat([df["Home_Id"], df["Away_Id"]]).unique()
    elo = {team: ELO_START for team in teams}  # elo-Werte der Teams
    #print(f"Initial Elo: {elo}")
    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        h = row["Home_Id"]
        a = row["Away_Id"]
        
        #print(f"Idx: {row.name}, Game: {row['Home']} vs {row['Away']}:Points: {row['Points']}")

        # Heimvorteil
        elo_home = elo[h] + HOME_ADV   # wird gleich 
        elo_away = elo[a]

        # Speichern für spätere Modelle
        elo_home_list.append(elo_home)   # wachsende Liste der Elo-Werte pro Spiel
        elo_away_list.append(elo_away)

        #print(f"elo_home_list: {elo_home_list}, elo_away_list: {elo_away_list}")
        #print(f"elo_home_list: {elo_home_list[-1]:.1f}, elo_away_list: {elo_away_list[-1]:.1f}")
        #print(f"Before: Elo Home: {elo[h]:.1f}, Elo Away: {elo[a]:.1f}")

        # Ergebnis in Elo-Scores umwandeln
        score_home, score_away = get_match_score(row["Points"], OT_WEIGHT)

        # Elo updaten
        new_home, new_away = update_elo(elo[h], elo[a], score_home, score_away, K)
        elo[h] = new_home
        elo[a] = new_away
        #print(f"After: Elo Home: {elo[h]:.1f}, Elo Away: {elo[a]:.1f}")
    

    elo_df = pd.DataFrame(list(elo.items()), columns=['Team_Id', 'Elo'])
    elo_df['Team_Name'] = elo_df['Team_Id'].apply(lambda x: Team.Name(x))
    elo_df = elo_df.sort_values('Elo', ascending=False)
    #print(elo_df[['Team_Name', 'Elo']].to_string(index=False))

    df["Elo_Home"] = elo_home_list
    df["Elo_Away"] = elo_away_list
    df["Elo_Diff"] = df["Elo_Home"] - df["Elo_Away"]

    return df

def create_elo_ranking(df, train_idx_start, train_idx_end) -> dict:
    elo: dict = {}
    for team_id in Team.AllIds():
        elo[team_id] = ELO_START

    GAMES_PER_ROUND = 7
    df_train = df.iloc[train_idx_start*GAMES_PER_ROUND:train_idx_end*GAMES_PER_ROUND] # nur Trainingsdaten

    for _, row in df_train.iterrows():
        h = row["Home_Id"]
        a = row["Away_Id"]
        
        #print(f"Idx: {row.name}, Game: {row['Home']} vs {row['Away']}:Points: {row['Points']}")

        # Heimvorteil
        elo_home = elo[h] + HOME_ADV   # beim predict dazu zählen
        elo_away = elo[a]
        
        # Ergebnis in Elo-Scores umwandeln
        score_home, score_away = get_match_score(row["Points"], OT_WEIGHT)

        # Elo updaten
        new_home, new_away = update_elo(elo[h], elo[a], score_home, score_away, K)
        elo[h] = new_home
        elo[a] = new_away
    return elo

def print_elo_ranking(elo):
    sorted_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    for team_id, elo_value in sorted_elo:
        print(f"{Team.Name(team_id)}: {elo_value:.1f}")

# ---------------------------------------------------------
# Beispiel: Modell trainieren + testen
# ---------------------------------------------------------
def run_elo_model(df, train_rounds=42, games_per_round=7):
    train_size = train_rounds * games_per_round

    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    # Elo auf Trainingsdaten berechnen
    df_train = compute_elo(df_train, K, HOME_ADV, OT_WEIGHT)

    # Elo für Testdaten weiterführen
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full = compute_elo(df_full, K, HOME_ADV, OT_WEIGHT)

    df_test = df_full.iloc[train_size:].copy()

    # Vorhersage
    df_test["Pred"] = predict_points(df_test, T1, T2)

    # df_full als csv speichern
    # df_full.to_csv("elo_full.csv", index=False)

    accuracy = (df_test["Pred"] == df_test["Points"]).mean()
    #df_test.to_csv("df_test.csv", index=False)

    return df_test, accuracy, df_full

# ---------------------------------------------------------
# Vorhersage
# ---------------------------------------------------------
def predict_points(df, THRESH1, THRESH2):
    preds = []
    for diff in df["Elo_Diff"]:
        if diff > THRESH1:
            preds.append(3)
        elif diff < THRESH2:
            preds.append(0)
        else:
            preds.append(1)
    return preds

def predict_game_outcome(elo_home, elo_away):
    """Vorhersage des Spielergebnisses basierend auf Elo-Differenz."""
    diff = elo_home - elo_away
    if diff > T1:
        return 3  # klarer Heimsieg
    elif diff < T2:
        return 0  # klarer Auswärtssieg
    else:
        return 1  # knapper Sieg (OT/SO)

def predict_game_outcome_df(elo_ranking: dict, X: pd.DataFrame) -> np.ndarray:
    y_hat = []
    for _, row in X.iterrows():
        home_team_id = row['Home_Id']
        away_team_id = row['Away_Id']
        
        elo_home = elo_ranking[home_team_id] + HOME_ADV
        elo_away = elo_ranking[away_team_id]

        outcome = predict_game_outcome(elo_home, elo_away)
        y_hat.append(outcome)
    return np.array(y_hat)

# ---------------------------------------------------------
# Train/Test
# ---------------------------------------------------------
def evaluate(df, K, HOME_ADV, OT_WEIGHT, THRESH1, THRESH2, train_games):
    df_train = df.iloc[:train_games].copy()
    df_test = df.iloc[train_games:].copy()

    df_train = compute_elo(df_train, K, HOME_ADV, OT_WEIGHT)

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full = compute_elo(df_full, K, HOME_ADV, OT_WEIGHT)

    df_test = df_full.iloc[train_games:].copy()
    df_test["Pred"] = predict_points(df_test, THRESH1, THRESH2)

    return (df_test["Pred"] == df_test["Points"]).mean()

def analyze_model_performance(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, 
                              train_idx_start: int, train_idx_end: int,
                              test_idx_start: int, test_idx_end: int):
    # df with features, y with target variable, idxs in rounds
    GAMES_PER_ROUND = 7
    test_df = df.iloc[test_idx_start*GAMES_PER_ROUND:test_idx_end*GAMES_PER_ROUND]
    y_test = y[test_idx_start*GAMES_PER_ROUND:test_idx_end*GAMES_PER_ROUND]
    print(f"Number of training items: {(train_idx_end- train_idx_start)*GAMES_PER_ROUND}, test items: {len(test_df)}")
    print(f"Training on rounds {train_idx_start} to {train_idx_end}, testing on rounds {test_idx_start} to {test_idx_end}")

    elo_ranking = create_elo_ranking(df, train_idx_start=train_idx_start, train_idx_end=train_idx_end)
    y_hut = predict_game_outcome_df(elo_ranking, test_df)

    dp.score(y_test, y_hut)
    #dp.score_last_10_games(y_test, y_hut) #only for test

# ---------------------------------------------------------
# Grid Search
# ---------------------------------------------------------
def optimize_elo(df, train_rounds=40, games_per_round=7):
    train_games = train_rounds * games_per_round

    K_values = [10, 15, 20, 25, 30]
    HOME_ADV_values = [10, 20, 30, 40, 50]
    OT_WEIGHT_values = [0.4, 0.5, 0.6, 0.7]
    THRESH1_values = [10, 20, 30, 40]
    THRESH2_values = [-10, -20, -30, -40]

    best_acc = 0
    best_params = None

    for K, HA, OTW, T1, T2 in product(
        K_values, HOME_ADV_values, OT_WEIGHT_values, THRESH1_values, THRESH2_values
    ):
        acc = evaluate(df, K, HA, OTW, T1, T2, train_games)

        if acc > best_acc:
            best_acc = acc
            best_params = (K, HA, OTW, T1, T2)

    return best_acc, best_params

# ---------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------
def plot_elo_for_team(df, team_id):
    team_name = Team.Name(team_id)
    df_team = df[(df["Home_Id"] == team_id) | (df["Away_Id"] == team_id)].copy()
    df_team["Elo"] = np.where(df_team["Home_Id"] == team_id, df_team["Elo_Home"], df_team["Elo_Away"])
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_team.index, df_team["Elo"], marker='o', label='Elo', alpha=0.7)
    plt.title(f"Elo-Verlauf für {team_name}")
    plt.xlabel("Spielindex")
    plt.ylabel("Elo-Wert")
    plt.legend()
    plt.grid()
    plt.show()
    print(f"Final Elo for {team_name}: {df_team['Elo'].iloc[-1]:.1f}")

def plot_expected_score():
    elo_diffs = np.arange(-400, 401, 10)
    expected_scores = [expected_score(1500 + diff, 1500) for diff in elo_diffs]
    plt.plot(elo_diffs, expected_scores)
    plt.title("Erwartete Gewinnwahrscheinlichkeit vs Elo-Differenz")
    plt.xlabel("Elo-Differenz (Home - Away)")
    plt.ylabel("Erwartete Gewinnwahrscheinlichkeit (Home)")
    plt.grid()
    plt.show()

def plot_update_elo():  # Elo-Update in Abhängigkeit von aktueller Stärke in Elo
    score = np.linspace(0, 1, 10)
    print(f"score: {score}")
    new_home = [K*(1-s) for s in score]
    print(f"score: {score}, new_home: {new_home}")
    plt.plot(score, new_home, label='New Home Elo')
    plt.title("Elo-Update vs Gewinnwahrscheinlichkeit")
    plt.xlabel("Gewinnwahrscheinlichkeit (Home)")
    plt.ylabel("Neuer Elo-Wert")
    plt.legend()
    plt.grid()
    plt.show()

def  plot_Elo_HC_Davos_Ajoie(df):
    df_test, acc, df_full = run_elo_model(df)
    plot_elo_for_team(df_full, Team.Id("HC Davos"))
    plot_elo_for_team(df_full, Team.Id("HC Ajoie"))


# ---------------------------------------------------------
# Parameter
# ---------------------------------------------------------
ELO_START = 1500
K = 15                 # Update speed
HOME_ADV = 0          # Heimvorteil in Elo-Punkten
OT_WEIGHT = 0.5        # OT/SO zählt als halber Sieg
T1 = 40                # Schwellenwert für klare Heimsiege
T2 = -10               # Schwellenwert für klare Auswärtssiege
# ---------------------------------------------------------

if __name__ == "__main__":
    X, y, df = dp.load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf 
    
    # Optimize Hyperparameters (K, HOME_ADV, OT_WEIGHT, T1, T2) mit Grid Search
    # best_acc, best_params = optimize_elo(df)
    # print("Beste Accuracy:", best_acc)
    # print("Beste Parameter:", best_params) # (K = 15, HA = 10, OTW = 0.5, T1 = 40, T2 = -10) 
    

    # Elo plots und Analyse
    plot_expected_score()
    plot_update_elo()
    i=0
    df_test, acc, df_full = run_elo_model(df)
    print("Accuracy:", acc) #0.44285714285714284
    print(df_test[["Home", "Away", "Resultat", "Points", "Pred", "Elo_Diff"]].head(10))

    plot_elo_for_team(df_full, Team.Id("HC Davos"))
    plot_elo_for_team(df_full, Team.Id("HC Ajoie"))
    
    # Test
    print("All team names:", Team.AllNames())
    train_df = df.iloc[0:1]  # only first game, for Ambri vs Kloten: 2
    print(train_df[["Home", "Away", "Resultat", "Points"]])
    elo_ranking=create_elo_ranking(train_df, 0, 1)  # Ambri +7.5 und Kloten -7.5
    print_elo_ranking(elo_ranking)