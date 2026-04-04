import math
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

path_prefix = '../' if os.getcwd().endswith('test') else ''  # direct test run vs pytest on repository
sys.path.append(path_prefix + 'src')

from team import Team
import data_service as dp

def test_data_processing_points(): # tests target variable points
    df = dp.load_data()
    df_cleaned = dp.prepare_data(df)
    df_with_target = dp.add_target_variables(df_cleaned)
    points = df_with_target[df_with_target['Home'] == 'EV Zug']['Points'].sum()
    away_games = len(df_with_target[df_with_target['Away'] == 'EV Zug'])
    adversaire_points = df_with_target[df_with_target['Away'] == 'EV Zug']['Points'].sum()
    points += away_games*3 - adversaire_points
    assert points == 75, f"Expected points sum for EV Zug to be 75, but got {points}"

def test_data_processing_goals(): # tests target variables Home_Goals and Away_Goals
    df = dp.load_data()
    df_cleaned = dp.prepare_data(df)
    df_with_target = dp.add_target_variables(df_cleaned)
    num_rows = len(df_with_target[df_with_target['Home'] == 'ZSC Lions'])
    num_rows += len(df_with_target[df_with_target['Away'] == 'ZSC Lions'])
    assert num_rows == 52, f"Expected number of games for ZSC Lions to be 52, but got {num_rows}"
    goals_sum = df_with_target[df_with_target['Home'] == 'ZSC Lions']['Home_Goals'].sum()
    goals_sum += df_with_target[df_with_target['Away'] == 'ZSC Lions']['Away_Goals'].sum()
    assert goals_sum == 147, f"Expected goal sum for ZSC Lions to be 147, but got {goals_sum}"

def test_team_class():
    Team.initialize(['EV Zug', 'ZSC Lions', 'HC Davos', 'HC Ajoie'])
    assert Team.Id('EV Zug') == 0, f"Expected ID for EV Zug to be 0, but got {Team.Id('EV Zug')}"
    assert Team.Id('ZSC Lions') == 1, f"Expected ID for ZSC Lions to be 1, but got {Team.Id('ZSC Lions')}"
    assert Team.Name(0) == 'EV Zug', f"Expected name for ID 0 to be EV Zug, but got {Team.Name(0)}"
    assert Team.Name(1) == 'ZSC Lions', f"Expected name for ID 1 to be ZSC Lions, but got {Team.Name(1)}"

def test_rmse():
    y = np.array([3, 3, 3, 3, 3, 3, 3, 0, 0, 2])
    y_hut = np.array([3, 3, 3, 3, 3, 3, 3, 0, 0, 1])
    rmse_value = dp.rmse(y, y_hut)
    expected_rmse = math.sqrt(1/10)   # one prediction is off by 1, so MSE is 0.1 
    assert rmse_value == expected_rmse, f"Expected RMSE to be {expected_rmse}, but got {rmse_value}"
    y_hut = np.array([3, 3, 3, 3, 3, 3, 3, 0, 0, 2])
    rmse_value = dp.rmse(y, y_hut)
    expected_rmse = 0.0 # perfect prediction
    assert rmse_value == expected_rmse, f"Expected RMSE to be {expected_rmse}, but got {rmse_value}"

def test_accuracy_score():
    y = np.array([3, 3, 3, 3, 3, 3, 3, 0, 0, 2])
    y_hut = np.array([3, 3, 3, 3, 3, 3, 3, 0, 0, 1])
    accuracy = accuracy_score(y.astype(int), y_hut.astype(int))
    expected_accuracy = 0.9 # one prediction is off
    assert accuracy == expected_accuracy, f"Expected accuracy to be {expected_accuracy}, but got {accuracy}"
    y_hut = np.array([3, 3, 3, 3, 3, 3, 3, 0, 0, 2])
    accuracy = accuracy_score(y.astype(int), y_hut.astype(int))
    expected_accuracy = 1.0 # perfect prediction
    assert accuracy == expected_accuracy, f"Expected accuracy to be {expected_accuracy}, but got {accuracy}"

def test_team_ranking():
    X, y, df = dp.load()
    ranking_df = dp.create_team_ranking(df, 0, 52)
    expected_ranking = ['HC Davos', 'Fribourg-Gottéron','Genève-Servette HC', 'ZSC Lions', 'HC Lugano', 'Lausanne HC', 
                        'SC Rapperswil-Jona Lakers', 'EV Zug', 'SC Bern', 'EHC Biel-Bienne','SCL Tigers', 'EHC Kloten',
                         'HC Ambri-Piotta','HC Ajoie']
    
    actual_ranking = ranking_df['Team'].tolist()
    assert actual_ranking == expected_ranking, f"Expected team ranking to be {expected_ranking}, but got {actual_ranking}"
 

