import os
import sys

path_prefix = '../' if os.getcwd().endswith('test') else ''  # direct test run vs pytest on repository
sys.path.append(path_prefix + 'src')

import elo_regression as el

def test_elo_update():
    elo_update = el.update_elo(1500, 1500, 1, 0, 20)
    expected_elo_update = (1510.0, 1490.0)  # Gewinner erhält 10 Punkte, Verlierer verliert 10 Punkte
    assert elo_update == expected_elo_update, f"Expected Elo update to be {expected_elo_update}, but got {elo_update}"

def test_expected_score():
    expected_score = el.expected_score(1500, 1500)
    expected_expected_score = 0.5  # Bei gleichen Elo-Werten beträgt die erwartete Gewinnwahrscheinlichkeit 50%
    assert expected_score == expected_expected_score, f"Expected expected score to be {expected_expected_score}, but got {expected_score}"

    expected_score = el.expected_score(1600, 1500)
    expected_expected_score = 1 / (1 + 10 ** ((1500 - 1600) / 400))
    assert expected_score == expected_expected_score, f"Expected expected score to be {expected_expected_score}, but got {expected_score}"