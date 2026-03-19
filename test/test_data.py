import math
import os
import sys

path_prefix = '../' if os.getcwd().endswith('test') else ''  # direct test run vs pytest on repository
sys.path.append(path_prefix + 'src')


from hockey_prediction_main import load_data, prepare_data, add_target_variable, Select_Features_Target, convert_team_names_to_numeric, create_id_to_team_mapping, name_to_id
def test_dummy():
    df = load_data()
    df_cleaned = prepare_data(df)
    df_with_target = add_target_variable(df_cleaned)
    target_sum = df_with_target[df_with_target['Home'] == 'EV Zug']['Target'].sum()
    assert target_sum == 43, f"Expected target sum for EV Zug to be 75, but got {target_sum}"
