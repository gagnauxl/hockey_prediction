#!/usr/bin/python
import pandas as pd

# own modules
from team import Team
import data_service as dp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


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

    model = RandomForestClassifier()
    model.fit(train_df, y_train)
    y_hut = model.predict(test_df)
    print(f"y_hut: {y_hut[:10]}")
    dp.score(y_test, y_hut)

    importances = model.feature_importances_
    plt.figure(figsize=(10, 5))
    plt.barh(X.columns, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import data_service as dp
    SVS = True
    X, y, df = dp.load(SVS=SVS)
    analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=10, test_idx_start=10, test_idx_end=20)