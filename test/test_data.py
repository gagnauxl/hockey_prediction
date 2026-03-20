import math
import os
import sys

path_prefix = '../' if os.getcwd().endswith('test') else ''  # direct test run vs pytest on repository
sys.path.append(path_prefix + 'src')

from hockey_prediction_main import add_target_variable, load_data, prepare_data
from team import Team

def test_dummy():
    df = load_data()
    df_cleaned = prepare_data(df)
    df_with_target = add_target_variable(df_cleaned)
    target_sum = df_with_target[df_with_target['Home'] == 'EV Zug']['Target'].sum()
    assert target_sum == 43, f"Expected target sum for EV Zug to be 75, but got {target_sum}"

def test_team_class():
    Team.initialize(['EV Zug', 'ZSC Lions', 'HC Davos', 'HC Ajoie'])
    assert Team.Id('EV Zug') == 0, f"Expected ID for EV Zug to be 0, but got {Team.Id('EV Zug')}"
    assert Team.Id('ZSC Lions') == 1, f"Expected ID for ZSC Lions to be 1, but got {Team.Id('ZSC Lions')}"
    assert Team.Name(0) == 'EV Zug', f"Expected name for ID 0 to be EV Zug, but got {Team.Name(0)}"
    assert Team.Name(1) == 'ZSC Lions', f"Expected name for ID 1 to be ZSC Lions, but got {Team.Name(1)}"
