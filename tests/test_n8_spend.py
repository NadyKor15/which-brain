import pytest

from n8.spend import SpendCap, SpendCapExceeded, nova_usd, polly_usd, transcribe_usd


def test_refuses_the_call_that_would_pass_the_ceiling(tmp_path):
    cap = SpendCap(tmp_path / "spend.jsonl", max_usd=1.0)
    cap.charge(0.6, "a")
    with pytest.raises(SpendCapExceeded):
        cap.charge(0.5, "b")
    assert cap.spent() == pytest.approx(0.6)  # refused call is not recorded


def test_ceiling_is_cumulative_across_runs(tmp_path):
    ledger = tmp_path / "spend.jsonl"
    SpendCap(ledger, max_usd=1.0).charge(0.9, "first run")
    with pytest.raises(SpendCapExceeded):
        SpendCap(ledger, max_usd=1.0).charge(0.2, "second run")


def test_prices_match_cost_md():
    assert nova_usd("pro", 1000, 1000) == pytest.approx(0.0008 + 0.0032)
    assert transcribe_usd(2) == pytest.approx(15 * 0.0004)  # 15 s minimum
    assert polly_usd(1_000_000) == pytest.approx(16)
