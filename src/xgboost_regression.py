# own modules
from team import Team
import data_service as dp

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# ELO-FUNKTIONEN
# ---------------------------------------------------------

def expected_score(elo_home, elo_away):
    return 1 / (1 + 10 ** ((elo_away - elo_home) / 400))

def update_elo(elo_home, elo_away, score_home, score_away, K):
    exp_home = expected_score(elo_home, elo_away)
    new_home = elo_home + K * (score_home - exp_home)
    new_away = elo_away + K * (score_away - (1 - exp_home))
    return new_home, new_away

def get_match_score(points, OT_WEIGHT=0.5):
    if points == 3:
        return 1, 0
    elif points == 2:
        return OT_WEIGHT, 1 - OT_WEIGHT
    elif points == 1:
        return 1 - OT_WEIGHT, OT_WEIGHT
    else:
        return 0, 1

def compute_elo(df, K=20, HOME_ADV=30, OT_WEIGHT=0.5):
    df = df.copy()
    teams = pd.concat([df["Home_Id"], df["Away_Id"]]).unique()
    elo = {team: 1500 for team in teams}

    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        h, a = row["Home_Id"], row["Away_Id"]

        elo_home = elo[h] + HOME_ADV
        elo_away = elo[a]

        elo_home_list.append(elo_home)
        elo_away_list.append(elo_away)

        score_home, score_away = get_match_score(row["Points"], OT_WEIGHT)
        new_home, new_away = update_elo(elo[h], elo[a], score_home, score_away, K)

        elo[h] = new_home
        elo[a] = new_away

    df["Elo_Home"] = elo_home_list
    df["Elo_Away"] = elo_away_list
    df["Elo_Diff"] = df["Elo_Home"] - df["Elo_Away"]

    return df

# ---------------------------------------------------------
# FORM-FEATURES (KORRIGIERT)
# ---------------------------------------------------------

def add_team_form_features(df, window=5):
    df = df.copy()
    df["Game_Index"] = np.arange(len(df))

    # Long format
    home = df[["Game_Index", "Home_Id", "Points", "Home_Goals", "Away_Goals"]].copy()
    home.rename(columns={
        "Home_Id": "Team_Id",
        "Points": "Team_Points",
        "Home_Goals": "Goals_For",
        "Away_Goals": "Goals_Against"
    }, inplace=True)

    away = df[["Game_Index", "Away_Id", "Points", "Home_Goals", "Away_Goals"]].copy()
    away.rename(columns={
        "Away_Id": "Team_Id",
        "Points": "Team_Points",
        "Home_Goals": "Goals_Against",
        "Away_Goals": "Goals_For"
    }, inplace=True)

    long_df = pd.concat([home, away], ignore_index=True)
    long_df.sort_values(["Team_Id", "Game_Index"], inplace=True)

    long_df["Goal_Diff"] = long_df["Goals_For"] - long_df["Goals_Against"]

    # transform statt apply → kein MultiIndex
    long_df["Form_Points"] = (
        long_df.groupby("Team_Id")["Team_Points"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    )

    long_df["Form_GD"] = (
        long_df.groupby("Team_Id")["Goal_Diff"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    )

    # Merge zurück
    home_form = long_df[["Game_Index", "Team_Id", "Form_Points", "Form_GD"]].rename(
        columns={
            "Team_Id": "Home_Id",
            "Form_Points": "Home_Form_Points",
            "Form_GD": "Home_Form_GD"
        }
    )

    away_form = long_df[["Game_Index", "Team_Id", "Form_Points", "Form_GD"]].rename(
        columns={
            "Team_Id": "Away_Id",
            "Form_Points": "Away_Form_Points",
            "Form_GD": "Away_Form_GD"
        }
    )

    df = df.merge(home_form, on=["Game_Index", "Home_Id"], how="left")
    df = df.merge(away_form, on=["Game_Index", "Away_Id"], how="left")

    df.drop(columns=["Game_Index"], inplace=True)

    return df

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def build_features(df):
    df = compute_elo(df)
    df = add_team_form_features(df)

    df[["Home_Form_Points", "Home_Form_GD",
        "Away_Form_Points", "Away_Form_GD"]] = df[[
            "Home_Form_Points", "Home_Form_GD",
            "Away_Form_Points", "Away_Form_GD"
        ]].fillna(0)

    feature_cols = [
        "Elo_Home", "Elo_Away", "Elo_Diff",
        "Home_Form_Points", "Home_Form_GD",
        "Away_Form_Points", "Away_Form_GD"
    ]

    X = df[feature_cols].values
    y = df["Points"].values

    return df, X, y, feature_cols

# ---------------------------------------------------------
# KLASSEN-MAPPING (4 Klassen!)
# ---------------------------------------------------------

def map_points_to_classes(y):
    mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    return np.array([mapping[v] for v in y]), mapping

def inverse_map_classes(pred_classes, mapping):
    inv_map = {v: k for k, v in mapping.items()}
    return np.array([inv_map[c] for c in pred_classes])

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train_xgb_model(df, train_rounds=40, games_per_round=7):
    df = df.sort_index().reset_index(drop=True)

    df_feat, X, y, feature_cols = build_features(df)
    y_cls, mapping = map_points_to_classes(y)

    train_games = train_rounds * games_per_round

    X_train, X_test = X[:train_games], X[train_games:]
    y_train, y_test = y_cls[:train_games], y_cls[train_games:]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    y_pred_cls = model.predict(X_test)
    y_pred_points = inverse_map_classes(y_pred_cls, mapping)
    y_test_points = inverse_map_classes(y_test, mapping)

    acc = accuracy_score(y_test_points, y_pred_points)

    df_test = df_feat.iloc[train_games:].copy()
    df_test["Pred_Points"] = y_pred_points

    return model, df_test, acc, feature_cols

# ---------------------------------------------------------
# ANWENDUNG
# ---------------------------------------------------------

# df = ...  # Dein DataFrame
# model, df_test, acc, feature_cols = train_xgb_model(df)
# print("Test-Accuracy:", acc)
# print(df_test[["Home", "Away", "Points", "Pred_Points", "Elo_Diff"]].head())

if __name__ == "__main__":
    X, y, df = dp.load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf 

    model, df_test, acc, feature_cols = train_xgb_model(df, train_rounds=40, games_per_round=7)
    print("Test-Accuracy:", acc)
    print(df_test[["Home", "Away", "Points", "Pred_Points", "Elo_Diff",
                "Home_Form_Points", "Away_Form_Points"]].head())
