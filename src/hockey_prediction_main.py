#!/usr/bin/python
# to handle datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# own modules
import data_service as dp
import elo_regression
from team import Team
import ranking_regression as rr
import lin_regression as lr
import knn_regression as knn
import elo_regression as elo

"""
Eishockey Vorhersagemodell: Ziel ist es, die Punkte des Heimteams in einem Spiel 
vorherzusagen (0, 1 oder 3), basierend auf den Team-IDs und der Rangliste der Teams.
Die Daten enthalten Informationen über die Spiele, einschließlich der IDs 
der Heim- und Auswärtsteams
Ich habe folgendes Modell:
     Home_Id  Away_Id                       Home                       Away Resultat OT/SO  Points 
354       10       11                 EHC Kloten                  HC Lugano      2:1    OT       2 
355        5        1                  ZSC Lions                   HC Davos      1:2   NaN       0 
356        6        4                     EV Zug  SC Rapperswil-Jona Lakers      2:3    OT       1 
357        9        2                    SC Bern          Fribourg-Gottéron      3:4   NaN       0 
358        1        7                   HC Davos            EHC Biel-Bienne      5:2   NaN       3 
359        3       10         Genève-Servette HC                 EHC Kloten      7:0   NaN       3 
360       11        6                  HC Lugano                     EV Zug      4:1   NaN       3 
361       12        8                Lausanne HC                   HC Ajoie      2:3    SO       1 
362        4        0  SC Rapperswil-Jona Lakers            HC Ambri-Piotta      6:2   NaN       3 
363       13        5                 SCL Tigers                  ZSC Lions      0:2   NaN       0 

Die Zielvariable ist "Points", also die Punkte, die das Heimteam in diesem Spiel erzielt hat (0, 1 oder 3).
Die Features sind "Home_Id" und "Away_Id", also die IDs der Heim- und Auswärtsteams.
Die Spalten "Home", "Away", "Resultat" und "OT/SO" sind nicht direkt als Features nutzbar, könnten aber in einem erweiterten Modell z.B. durch One-Hot-Encoding oder durch Extraktion von Informationen (z.B. Anzahl Tore) genutzt werden.

Die Modelle für eine Vorhersage könnten wie folgt aussehen:
1. Lineare Regression: Vorhersage der Punkte basierend auf den Team-IDs:Accuracy ist ca. 0.4
2. KNN-Regression: Vorhersage der Punkte basierend auf den Team-IDs und 
den Punkten der vorherigen Spiele: Accuracy ist mit k=1 ca. 0.4, höhere k-Werte bringen nichts
3. Ranking-basiertes Modell: Vorhersage der Punkte basierend auf der Rangliste der Teams, 
die aus den vorherigen Spielen erstellt wird: Accuracy ist ca. 0.52, immerhin besser als 50%

Die Modelle werden auf den ersten 40 Runden (280 Spielen) trainiert und dann 
auf den nächsten 12 Runden getestet, um die Performance zu bewerten.
"""

if __name__ == "__main__":
     """
     Todo:
     - Done: dynamic train-test split basierend auf Runden, z.B. 10 Runden trainieren, dann testen, dann nächsten 10 Runden trainieren, etc.
     - Done: knn-fit, geht nicht und Neighborhoods > 1 führen zu schlechteren Ergebnissen, warum?
     - Rejected: signifikante Features identifizieren, z.B. Zuschauerzahl
     - Done Cross-Validation, teilweise gemacht, mit Train-Test-Split, aber könnte verbessert werden
     - Done: ELO-ähnliches Ranking der Teams, basierend auf den bisherigen Spielen, und dieses als Feature nutzen
     - ELO aber nur die letzten 10 Spiele berücksichtigen (nur Streak), damit es dynamisch bleibt, und nicht die gesamte Historie

     """
     X, y, df = dp.load()         # Lädt die Daten, bereitet sie vor und teilt sie in Features (X) und Zielvariable (y) auf   
     #print(f"Last 10 values of df: \n{df[['Home_Id', 'Away_Id', 'Home', 'Away', 'Resultat', 'OT/SO', 'Points']].tail(5).to_string()}")
     print(f"Last 10 values of df: \n{df[['Home_Id', 'Away_Id', 'Home', 'Away', 'Resultat', 'OT/SO', 'Points']].iloc[130:140].to_string()}")
     print(f"Features: {X.shape}")
     #X.info()
     print(X.head())
     print(f"\nShape of target variable y: {y.shape}\n{y[-5:]}")

     # train 40 Runden, test letzte 12 Runden
     #lr.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=40, test_idx_start=40, test_idx_end=52)
     # train 10 Runden, test nächste 10 Runden
     #lr.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=10, test_idx_start=10, test_idx_end=20)
     
     # train 40 Runden, test letzte 12 Runden
     #rr.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=40, test_idx_start=40, test_idx_end=52)
     # train 10 Runden, test nächste 10 Runden
     #rr.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=10, test_idx_start=10, test_idx_end=20)
 
     # train 40 Runden, test letzte 12 Runden
     #knn.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=40, test_idx_start=40, test_idx_end=52)
     # train 10 Runden, test nächste 10 Runden
     #knn.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=10, test_idx_start=10, test_idx_end=20)
     
     # train 52 Runden, test nächste 10 Runden, z.B. die letzten 10 Runden, um die Performance auf den letzten Spielen zu bewerten
     #knn.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=52, test_idx_start=42, test_idx_end=52)

     elo.plot_expected_score()
     elo.plot_update_elo()
     elo.analyze_model_performance(df=df, X=X, y=y, train_idx_start=0, train_idx_end=42, test_idx_start=42, test_idx_end=52)
     elo.plot_Elo_HC_Davos_Ajoie(df)
     
     print("end of main")
