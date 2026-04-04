#!/usr/bin/python
from xml.parsers.expat import model

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# own modules
from team import Team
import data_service as dp

def fit(X: pd.DataFrame, y: pd.Series) -> KNeighborsClassifier:
    model_knn  = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X, y)
    return model_knn

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
    y_hut = model.predict(test_df)

    dp.score(y_test, y_hut)
    #dp.score_last_10_games(y_test, y_hut) #just for testing

def best_knn(df: pd.DataFrame, y: pd.Series, train_games: int=40*7):  
    # df with features, y with target variable, train_game: 40*7 = 40 Runden
    train_df = df.iloc[:train_games]
    y_train = y.iloc[:train_games]

    from sklearn.model_selection import RepeatedKFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt

    # Pipeline: Skalieren + kNN
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Grid wie in R: k = 4:12
    param_grid = {
        'knn__n_neighbors': list(range(1, 3))
    }

    # Cross‑Validation: 10‑fold, 5 Wiederholungen
    # cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)

    cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)

    # Grid Search
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy'
    )

    grid.fit(train_df, y_train)

    print("Bestes k:", grid.best_params_)
    print("Beste Accuracy:", grid.best_score_)

    # Plot analog zu plot(knn_fit)
    results = grid.cv_results_
    k_values = results['param_knn__n_neighbors']
    mean_scores = results['mean_test_score']

    plt.plot(k_values, mean_scores, marker='o')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("kNN Cross‑Validation Accuracy")
    plt.grid(True)
    plt.show()

    return grid.best_params_['knn__n_neighbors']

if __name__ == "__main__":
    X, y, df = dp.load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf   
    model_knn  = KNeighborsClassifier(n_neighbors=1)
    model_knn.fit(X, y)          # Trainiert auf ALLE Daten

    # was ist der Unterschied von df zu X:
    print(f"\nShape of dataframe X (rows, columns): {X.shape}")
    X.info()
    print(X.head())
    print(f"\nShape of dataframe df (rows, columns): {df.shape}")
    df.info()
    print(df.head())

    # best_knn(df, y) # funktioniert nicht, da es zu wenig Daten gibt, um k > 1 zu testen, siehe Fehlermeldung: ValueError: n_splits=3 cannot be greater than the number of members in each class.

    y = y[0:10].to_numpy()
    y_hut = model_knn.predict(X[0:10])

    print(f"first 10 games:\n{df[0:10]}")
    print(f"Actual outcomes for the first 10 games: {y}")
    print(f"                             Predicted: {y_hut}")

    y_model_knn = model_knn.predict(X)
    accuracy = accuracy_score(y.astype(int), y_hut.astype(int))
    print(f"KNN Accuracy: {accuracy}")

    print("Specific game predictions:")
    print("Specific game predictions:")
    print(f"ZSC Lions - EV Zug: {model_knn.predict(pd.DataFrame([[Team.Id('ZSC Lions'), Team.Id('EV Zug')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"EV Zug - ZSC Lions: {model_knn.predict(pd.DataFrame([[Team.Id('EV Zug'), Team.Id('ZSC Lions')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"HC Davos - HC Ajoie: {model_knn.predict(pd.DataFrame([[Team.Id('HC Davos'), Team.Id('HC Ajoie')]], columns=['Home_Id', 'Away_Id']))}")
    print(f"HC Ajoie - HC Davos: {model_knn.predict(pd.DataFrame([[Team.Id('HC Ajoie'), Team.Id('HC Davos')]], columns=['Home_Id', 'Away_Id']))}")