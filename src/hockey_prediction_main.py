#!/usr/bin/python
# to handle datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# own modules
import data_service as dp
from team import Team
import ranking_regression as rr
import lin_regression as lr

#------- Modeling functions -------
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

if __name__ == "__main__":
    df = dp.load_data()
    dp.define_team_ids(df)     # Initialisiert die Team-IDs basierend auf den Daten, damit sie in der Team-Klasse verfügbar sind
     
    df = dp.prepare_data(df)  # Bereinigt die Daten, z.B. durch Entfernen unnötiger Spalten, nur noch Features
    df = dp.add_team_ids(df)   # Konvertiert die Teamnamen in numerische IDs, damit sie für das Modell verwendet werden können

    df = dp.add_target_variables(df) # Fügt die Zielvariable hinzu, z.B. durch Berechnung des Spielausgangs basierend auf den Ergebnissen
    

    X, y = dp.Select_Features_Target(df)  # Wählt die relevanten Features (Home, Away) und die Zielvariable (Target) aus, um sie für das Modelltraining vorzubereiten
    
    model = fit(X, y)          # Trainiert das Modell mit den ausgewählten Features und der Zielvariable
    dp.plot_results(df, model)    # Hier könnte man die tatsächlichen Ergebnisse vs. die vorhergesagten Ergebnisse plotten, um die Leistung des Modells zu visualisieren   
    y_hut = predict(model, X)
    rmse_value = dp.rmse(y, y_hut)
    print(f"Model RMSE: {rmse_value}, y_hut: {y_hut[:10]}")
    print(f"Model R^2: {dp.R_squared(y.to_numpy(), y_hut)}")


    y_hut_approx = predict_rounded(y_hut)
    rmse_value = dp.rmse(y, y_hut_approx)
    print(f"Model RMSE Rounded: {rmse_value}, y_hut_approx: {y_hut_approx[:10]}")
    print(f"Model R^2 Rounded: {dp.R_squared(y.to_numpy(), y_hut_approx)}")
    accuracy = accuracy_score(y.astype(int), y_hut_approx.astype(int))
    print(f"Accuracy: {accuracy}")
    
    y_hut_1_7 = np.full_like(y_hut, 1.7, dtype=float)
    rmse_value = dp.rmse(y, y_hut_1_7)
    print(f"Model RMSE Approximated 1.7: {rmse_value}, y_hut_1_7: {y_hut_1_7[:10]}")

    ranking_df = dp.create_team_ranking(df, 26)
    y_hut_based_on_ranking = rr.predict_game_outcome_custom(ranking_df, X)
    rmse_value = dp.rmse(y, y_hut_based_on_ranking)
    print(f"Model RMSE based on ranking: {rmse_value}, y_hut_based_on_ranking: {y_hut_based_on_ranking[:10]}")
    accuracy = accuracy_score(y.astype(int), y_hut_based_on_ranking.astype(int))
    print(f"Accuracy: {accuracy}")

    y_hut_based_on_ranking_rounded = predict_rounded(y_hut_based_on_ranking)
    rmse_value = dp.rmse(y, y_hut_based_on_ranking_rounded)
    print(f"Model RMSE based on ranking Rounded: {rmse_value}, y_hut_based_on_ranking_rounded: {y_hut_based_on_ranking_rounded[:10]}")
    accuracy = accuracy_score(y.astype(int), y_hut_based_on_ranking.astype(int))
    print(f"Accuracy: {accuracy}")

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
    print(f"Linear Regression R^2: {dp.R_squared(y.to_numpy(), y_model)}")

    y_model_knn = model_knn.predict(X)
    accuracy = accuracy_score(y.astype(int), y_model_knn.astype(int))
    print(f"KNN Accuracy: {accuracy}")