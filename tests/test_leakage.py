import os
import pytest
from engine.data_loader import load_ff25_daily
from engine.backtester import cross_sectional_ls
from factors.program import evaluate_program
import json

DATA_PATH = "data/ff25_daily.csv"

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Data file not present; skipping leakage test.")
def test_leakage_flag_changes_with_delay():
    df = load_ff25_daily(DATA_PATH).iloc[:400]
    prog = json.loads(open("factors/baseline_program.json","r").read())
    scores = evaluate_program(prog, df)

    out0 = cross_sectional_ls(df, scores, delay_days=0)
    out1 = cross_sectional_ls(df, scores, delay_days=1)

    assert out0["leakage_flag"] is True
    assert out1["leakage_flag"] is False
