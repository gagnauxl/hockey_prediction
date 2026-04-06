#!/usr/bin/python
import pandas as pd

# own modules
from team import Team
import data_service as dp
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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

    model = DecisionTreeRegressor()
    model.fit(train_df, y_train)
    y_hut = model.predict(test_df)

    dp.score(y_test, y_hut)

    print(f"Decision Tree")
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.show()

    #dp.score_last_10_games(y_test, y_hut) #just for testing