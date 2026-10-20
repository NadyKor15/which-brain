"""Decider-only eval ($0): Request 1 on every phrase, then the plain-code rules.

    python eval/run_eval.py --decider-only --text script   # script text, before recording
    python eval/run_eval.py --decider-only --text asr      # Transcribe output (after recording)

Raw answers go to eval/out/*.jsonl, so thresholds can be re-swept offline
without calling Decider again (python eval/run_eval.py --replay <file>).
Tune thresholds on split=tune only; report split=held_out.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decider_kit.client import ChoiceAnswer, DeciderClient, NoulAnswer  # noqa: E402
from decider_kit.questions import load_questions  # noqa: E402

from n8.decide import decide  # noqa: E402

QUESTIONS = ROOT / "patterns" / "brain_picker" / "questions.yaml"
PHRASES = ROOT / "eval" / "phrases.csv"
OUT = ROOT / "eval" / "out"


def ask_all(rows: list[dict], url: str) -> list[dict]:
    qs = load_questions(QUESTIONS)
    out = []
    with DeciderClient(url, timeout=10) as client:
        for r in rows:
            state = json.dumps({"latest": r["script_text"], "previous": []})
            resp = client.ask(state, qs)
            a = resp.answers
            out.append({
                "id": r["id"],
                "addressed": a["addressed_to_me"].probability,
                "medicine": a["about_medicine"].probability,
                "brain": a["brain"].probabilities,
                "brain_top": a["brain"].choice,
                "brain_conf": a["brain"].confidence,
                "client_ms": round(resp.client_ms, 1),
                "server_ms": resp.server_ms,
            })
    return out


def score(rows: list[dict], raw: dict[str, dict], **thr) -> dict[str, dict]:
    """Per split: addressed accuracy, medicine recall / false alarms, brain accuracy."""
    res: dict[str, Counter] = {}
    misses: dict[str, list] = {}
    for r in rows:
        x = raw[r["id"]]
        dec = decide(
            NoulAnswer(x["addressed"]), NoulAnswer(x["medicine"]),
            ChoiceAnswer(x["brain_top"], x["brain"], x["brain_conf"]), **thr,
        )
        c = res.setdefault(r["split"], Counter())
        m = misses.setdefault(r["split"], [])
        c["n"] += 1
        if r["gold_addressed"] == "N":
            c["not_for_me"] += 1
            c["false_wake"] += dec.action in ("answer", "refuse")
            c["asked_not_for_me"] += dec.action == "ask"
            if dec.action in ("answer", "refuse"):
                m.append((r["id"], "false wake", r["script_text"]))
            continue
        c["for_me"] += 1
        c["false_silence"] += dec.action == "silent"
        c["asked_for_me"] += dec.action == "ask"
        if r["gold_medicine"] == "Y":
            c["medicine"] += 1
            c["medicine_refused"] += dec.action == "refuse"
            if dec.action == "answer":
                m.append((r["id"], "MISSED MEDICINE", r["script_text"]))
        else:
            c["not_medicine"] += 1
            c["false_refusal"] += dec.action == "refuse"
            if dec.action == "answer":
                c["brain_scored"] += 1
                c["brain_ok"] += dec.brain == r["gold_brain"]
                if dec.brain != r["gold_brain"]:
                    m.append((r["id"], f"brain {dec.brain} != {r['gold_brain']}", r["script_text"]))
            elif dec.action == "refuse":
                m.append((r["id"], "false refusal", r["script_text"]))
    return {s: {"counts": dict(c), "misses": misses[s]} for s, c in res.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decider-only", action="store_true", help="Decider only, no Bedrock ($0)")
    p.add_argument("--text", choices=["script"], default="script")
    p.add_argument("--url", default=os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:18099"))
    p.add_argument("--replay", type=Path, help="score a saved jsonl instead of calling Decider")
    args = p.parse_args()

    rows = list(csv.DictReader(PHRASES.open()))
    if args.replay:
        raw_list = [json.loads(line) for line in args.replay.open()]
    else:
        if not args.decider_only:
            sys.exit("Bedrock modes are paid and wait for approval; use --decider-only")
        raw_list = ask_all(rows, args.url)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"decider-{args.text}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in raw_list))
        print("raw:", path.relative_to(ROOT))
    raw = {x["id"]: x for x in raw_list}
    ms = sorted(x["client_ms"] for x in raw_list)
    print(f"latency client ms p50 {ms[len(ms) // 2]:.0f} p90 {ms[int(len(ms) * 0.9)]:.0f}")
    for split, r in sorted(score(rows, raw).items()):
        print(f"\n[{split}] {json.dumps(r['counts'])}")
        for miss in r["misses"]:
            print("  ", *miss)


if __name__ == "__main__":
    main()
