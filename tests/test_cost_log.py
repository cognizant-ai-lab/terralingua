"""costs.csv carries the running total cost, continues it across a resume, and
gains the column when an older file lacks it."""

import csv

from terralingua.experiment.runner import SimulationRunner


def make_runner(tmp_path):
    runner = SimulationRunner.__new__(SimulationRunner)
    runner._cost_csv_path = tmp_path / "costs.csv"
    runner._cost_total = runner._resume_cost_total()
    return runner


def rows(tmp_path):
    with open(tmp_path / "costs.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_total_column_accumulates(tmp_path):
    runner = make_runner(tmp_path)
    runner._log_step_usage(0, {"a": {"cost": 0.5, "input_tokens": 10, "output_tokens": 1}})
    runner._log_step_usage(1, {"a": {"cost": 0.25, "input_tokens": 5, "output_tokens": 2}, "b": {"cost": 0.5}})
    recorded = rows(tmp_path)
    assert [r["total_cost_usd"] for r in recorded] == ["0.500000", "1.250000"]
    assert [r["cost_usd"] for r in recorded] == ["0.500000", "0.750000"]
    assert recorded[1]["input_tokens"] == "5" and recorded[1]["output_tokens"] == "2"


def test_total_continues_across_a_resume(tmp_path):
    make_runner(tmp_path)._log_step_usage(0, {"a": {"cost": 2.0}})
    resumed = make_runner(tmp_path)
    assert resumed._cost_total == 2.0
    resumed._log_step_usage(1, {"a": {"cost": 1.0}})
    assert [r["total_cost_usd"] for r in rows(tmp_path)] == ["2.000000", "3.000000"]


def test_file_without_the_column_gets_it(tmp_path):
    (tmp_path / "costs.csv").write_text(
        "timestep,cost_usd,input_tokens,output_tokens\n0,0.100000,1,1\n1,0.200000,2,2\n"
    )
    runner = make_runner(tmp_path)
    assert round(runner._cost_total, 6) == 0.3
    runner._log_step_usage(2, {"a": {"cost": 0.4}})
    recorded = rows(tmp_path)
    assert [r["total_cost_usd"] for r in recorded] == ["0.100000", "0.300000", "0.700000"]
    assert [r["timestep"] for r in recorded] == ["0", "1", "2"]


def test_cache_counts_survive_resume_without_fabricating_historical_hits(tmp_path):
    (tmp_path / "costs.csv").write_text(
        "timestep,cost_usd,input_tokens,output_tokens,total_cost_usd\n"
        "0,0.5,1200,10,0.500000\n"
    )
    runner = make_runner(tmp_path)
    runner._log_step_usage(
        1,
        {
            "a": {
                "input_tokens": 1500,
                "cached_input_tokens": 1024,
                "cache_write_input_tokens": 100,
            },
            "b": {
                "input_tokens": 2000,
                "cached_input_tokens": 1536,
                "cache_write_input_tokens": 0,
            },
        },
    )
    recorded = rows(tmp_path)
    assert recorded[0]["cached_input_tokens"] == ""
    assert recorded[0]["cache_write_input_tokens"] == ""
    assert recorded[1]["cached_input_tokens"] == "2560"
    assert recorded[1]["cache_write_input_tokens"] == "100"
    runner._log_step_usage(2, {"a": {"input_tokens": 10}})
    assert rows(tmp_path)[2]["cached_input_tokens"] == ""
