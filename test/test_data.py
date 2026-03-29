import math
import os
import sys

path_prefix = '../' if os.getcwd().endswith('test') else ''  # direct test run vs pytest on repository
sys.path.append(path_prefix + 'src')

from hockey_prediction_main import add_target_variables, load_data, prepare_data
from team import Team

def test_data_processing():
    df = load_data()
    df_cleaned = prepare_data(df)
    df_with_target = add_target_variables(df_cleaned)
    points = df_with_target[df_with_target['Home'] == 'EV Zug']['Points'].sum()
    away_games = len(df_with_target[df_with_target['Away'] == 'EV Zug'])
    adversaire_points = df_with_target[df_with_target['Away'] == 'EV Zug']['Points'].sum()
    points += away_games*3 - adversaire_points
    assert points == 75, f"Expected points sum for EV Zug to be 75, but got {points}"

def test_team_class():
    Team.initialize(['EV Zug', 'ZSC Lions', 'HC Davos', 'HC Ajoie'])
    assert Team.Id('EV Zug') == 0, f"Expected ID for EV Zug to be 0, but got {Team.Id('EV Zug')}"
    assert Team.Id('ZSC Lions') == 1, f"Expected ID for ZSC Lions to be 1, but got {Team.Id('ZSC Lions')}"
    assert Team.Name(0) == 'EV Zug', f"Expected name for ID 0 to be EV Zug, but got {Team.Name(0)}"
    assert Team.Name(1) == 'ZSC Lions', f"Expected name for ID 1 to be ZSC Lions, but got {Team.Name(1)}"

def test_data_processing_2():
    df = load_data()
    df_cleaned = prepare_data(df)
    df_with_target = add_target_variables(df_cleaned)
    num_rows = len(df_with_target[df_with_target['Home'] == 'ZSC Lions'])
    num_rows += len(df_with_target[df_with_target['Away'] == 'ZSC Lions'])
    assert num_rows == 52, f"Expected number of games for ZSC Lions to be 52, but got {num_rows}"
    goals_sum = df_with_target[df_with_target['Home'] == 'ZSC Lions']['Home_Goals'].sum()
    goals_sum += df_with_target[df_with_target['Away'] == 'ZSC Lions']['Away_Goals'].sum()
    assert goals_sum == 147, f"Expected goal sum for ZSC Lions to be 147, but got {goals_sum}"

