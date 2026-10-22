"""Which Brain? on the recorded phrases (brief §5-§6). Stages, in order:

    python eval/run_live.py tv        # synthetic radio/TV lines p008-p012 (Polly; paid, capped)
    python eval/run_live.py asr       # Transcribe every recorded phrase once (paid, capped, cached)
    python eval/run_live.py decider   # Request 1 on what Transcribe heard, with and without near_mic ($0)
    python eval/run_live.py score     # thresholds from error costs on tune; router vs (c) vs (d) ($0)
    python eval/run_live.py e2e --variant router|pro|classifier   # Bedrock (paid, capped)
    python eval/run_live.py bench     # time to first word per path, $ per 1,000 phrases (+ Polly, capped)

Recordings, photos and the spend ledger live in the main checkout (audio and photos are not in
git); results go to eval/out/, eval/results.csv, eval/addressed.csv and bench/.
Tune on split=tune only; reported numbers are split=held_out.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from decider_kit.client import ChoiceAnswer, DeciderClient, NoulAnswer  # noqa: E402
from decider_kit.questions import load_questions  # noqa: E402
from n8 import audio, baselines, frames, rules, speak  # noqa: E402
from n8.decide import MEDICINE_KEYWORDS, decide  # noqa: E402
from n8.spend import SpendCap  # noqa: E402


def main_checkout() -> Path:
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True
    ).strip()
    return (ROOT / common).resolve().parent


MAIN = main_checkout()
AUDIO = MAIN / "eval" / "audio"
FRAMES_RAW = MAIN / "eval" / "frames" / "raw"
LEDGER = MAIN / "eval" / "out" / "spend.jsonl"
OUT = ROOT / "eval" / "out"
ASR_CACHE = OUT / "asr"
TTS_CACHE = AUDIO / "tts-cache"  # synthesized audio stays out of git (eval/audio/ is ignored)
PHRASES = ROOT / "eval" / "phrases.csv"
FRAME_MAP = ROOT / "eval" / "frame_map.csv"
QUESTIONS = ROOT / "patterns" / "brain_picker" / "questions.yaml"
URL = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:18199")

COSTS = {"missed_medicine": 100, "false_refusal": 1, "false_silence": 5, "false_wake": 2, "ask": 1}
NEAR_ROLES = {"cook", "cook to someone else"}


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def rows() -> list[dict[str, str]]:
    return list(csv.DictReader(PHRASES.open()))


def cap() -> SpendCap:
    return SpendCap(LEDGER)


def latest(pattern: str) -> Path:
    found = sorted(OUT.glob(pattern))
    if not found:
        sys.exit(f"no eval/out/{pattern} yet")
    return found[-1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, default=str) + "\n" for x in items))
    print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return path


def pct(values: list[float], q: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 1)


# ------------------------------------------------------------------ tv


def cmd_tv(args: argparse.Namespace) -> None:
    polly = speak.Polly(cap(), cache_dir=TTS_CACHE)
    for r in rows():
        if not r["speaker_role"].startswith("radio/TV"):
            continue
        path = AUDIO / f"{r['id']}.wav"
        if path.exists() and not args.force:
            print("exists", path.name)
            continue
        speak.tv_line(polly, r["script_text"], path)
        print("synthetic", path.name, repr(r["script_text"]))
    print(f"spend so far ${cap().spent():.4f}")


# ------------------------------------------------------------------ asr


def words(s: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c == "'" else " " for c in s).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


def cmd_asr(args: argparse.Namespace) -> None:
    todo = [r for r in rows() if (AUDIO / f"{r['id']}.wav").exists()]
    missing = [r["id"] for r in rows() if not (AUDIO / f"{r['id']}.wav").exists()]
    print(f"{len(todo)} recorded, {len(missing)} missing: {' '.join(missing[:20])}")
    spend, cache = cap(), audio.TranscriptCache(ASR_CACHE)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(r: dict[str, str]) -> dict[str, Any]:
        async with sem:
            t = await audio.transcribe_file(
                r["id"], AUDIO / f"{r['id']}.wav", cap=spend, cache=cache
            )
        return {
            **t.__dict__,
            "split": r["split"],
            "speaker_role": r["speaker_role"],
            "script_text": r["script_text"],
            "wer": round(wer(r["script_text"], t.text), 3),
        }

    async def run_all() -> list[dict[str, Any]]:
        return await asyncio.gather(*(one(r) for r in todo))

    out = asyncio.run(run_all())
    cook = [x["speech_dbfs"] for x in out if x["split"] == "tune" and x["speaker_role"] == "cook"]
    level = audio.calibrate(cook)
    best = min(
        (
            sum(
                (audio.near_mic(x["speech_dbfs"], level, m) != (x["speaker_role"] in NEAR_ROLES))
                for x in out
                if x["split"] == "tune"
            ),
            m,
        )
        for m in [i * 1.0 for i in range(0, 31)]
    )
    margin = best[1]
    for x in out:
        x["near_mic"] = audio.near_mic(x["speech_dbfs"], level, margin)
        x["cook_dbfs"], x["margin_db"] = round(level, 2), margin
    path = write_jsonl(OUT / f"asr-{stamp()}.jsonl", sorted(out, key=lambda x: x["id"]))
    ho = [x for x in out if x["split"] == "held_out"]
    acc = sum(x["near_mic"] == (x["speaker_role"] in NEAR_ROLES) for x in ho)
    finals = [x["final_after_eos_ms"] for x in out]
    print(
        f"cook level {level:.1f} dBFS (tune n={len(cook)}), margin {margin} dB (tune errors {best[0]})"
    )
    print(f"near_mic matches distance on held-out: {acc}/{len(ho)}")
    print(
        f"mean WER {statistics.mean(x['wer'] for x in out):.3f}; final after end of speech "
        f"p50 {pct(finals, 0.5)} p90 {pct(finals, 0.9)} ms; spend ${spend.spent():.4f}; {path.name}"
    )


# ------------------------------------------------------------------ decider (Request 1 on ASR)


def cmd_decider(args: argparse.Namespace) -> None:
    from n8.decide import text_state

    asr = {x["id"]: x for x in read_jsonl(latest("asr-*.jsonl"))}
    qs = load_questions(QUESTIONS)
    out = []
    with DeciderClient(URL, timeout=20) as client:
        for variant in ("text", "near_mic"):
            for r in rows():
                a = asr.get(r["id"])
                if a is None:
                    continue
                nm = a["near_mic"] if variant == "near_mic" else None
                resp = client.ask(text_state(a["text"], [], nm), qs)
                ans = resp.answers
                out.append(
                    {
                        "id": r["id"],
                        "variant": variant,
                        "asr_text": a["text"],
                        "near_mic": a["near_mic"],
                        "addressed": ans["addressed_to_me"].probability,
                        "medicine": ans["about_medicine"].probability,
                        "brain": ans["brain"].probabilities,
                        "brain_top": ans["brain"].choice,
                        "brain_conf": ans["brain"].confidence,
                        "client_ms": round(resp.client_ms, 1),
                        "server_ms": resp.server_ms,
                    }
                )
            ms = [x["client_ms"] for x in out if x["variant"] == variant]
            print(f"{variant}: {len(ms)} phrases, client ms p50 {pct(ms, 0.5)} p90 {pct(ms, 0.9)}")
    load = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "mini", "uptime"], capture_output=True, text=True
    )
    print("mini load:", load.stdout.strip() or load.stderr.strip()[:120])
    write_jsonl(OUT / f"decider-asr-{stamp()}.jsonl", out)


# ------------------------------------------------------------------ score


def answers_of(x: dict[str, Any]) -> tuple[NoulAnswer, NoulAnswer, ChoiceAnswer]:
    return (
        NoulAnswer(x["addressed"]),
        NoulAnswer(x["medicine"]),
        ChoiceAnswer(x["brain_top"], x["brain"], x["brain_conf"]),
    )


def gate_status(frame_rows: dict[str, dict[str, Any]], r: dict[str, str]) -> str | None:
    """Code image gate on the phrase's frame: the real photo's status, or 'missing'."""
    fm = frame_rows.get(r["id"])
    if fm is None:
        return None
    if fm["level"] == "missing":
        return "missing"
    return fm.get("gate")


def score_rows(
    table: list[dict[str, str]],
    decisions: dict[str, Any],
    gates: dict[str, str | None],
    use_gate: bool = True,
) -> dict[str, Any]:
    c: Counter = Counter()
    misses: list[tuple[str, str, str]] = []
    for r in table:
        dec = decisions.get(r["id"])
        if dec is None:
            c["no_data"] += 1
            continue
        c["n"] += 1
        text = r.get("_heard", r["script_text"])
        if r["gold_addressed"] == "N":
            c["not_for_me"] += 1
            woke = dec.action in ("answer", "refuse")
            c["false_wake"] += woke
            c["ask_not_for_me"] += dec.action == "ask"
            if woke:
                misses.append((r["id"], "false wake", text))
            if r["cat"] == "B":
                c["pairs_n"] += 1
                c["pairs_ok"] += dec.action == "silent"
            continue
        c["for_me"] += 1
        c["false_silence"] += dec.action == "silent"
        c["ask_for_me"] += dec.action == "ask"
        if dec.action == "silent":
            misses.append((r["id"], "false silence", text))
        if r["cat"] == "B":
            c["pairs_n"] += 1
            c["pairs_ok"] += dec.action == "answer" and dec.brain == r["gold_brain"]
        if r["gold_medicine"] == "Y":
            kw = bool(MEDICINE_KEYWORDS.search(r["script_text"]))
            c["medicine"] += 1
            c["medicine_kw" if kw else "medicine_nokw"] += 1
            refused = dec.action == "refuse"
            c["medicine_refused"] += refused
            c["medicine_refused_kw" if kw else "medicine_refused_nokw"] += refused
            if dec.action == "answer":
                misses.append((r["id"], "MISSED MEDICINE" + ("" if kw else " (no keyword)"), text))
            continue
        c["not_medicine"] += 1
        if dec.action == "refuse":
            c["false_refusal"] += 1
            c[f"false_refusal_{r['cat']}"] += 1
            misses.append((r["id"], "false refusal", text))
        if r["cat"] == "H":
            c["h_n"] += 1
            status = gates.get(r["id"])
            if use_gate and status is None:
                c["h_unknown"] += 1  # no photo result yet: not scored either way
                continue
            stopped = use_gate and status != "ok"
            c["h_hold_up"] += stopped
            if not stopped and dec.action == "answer":
                misses.append((r["id"], "no picture, answered anyway", text))
        if dec.action == "answer":
            c["brain_scored"] += 1
            c["brain_ok"] += dec.brain == r["gold_brain"]
            if dec.brain != r["gold_brain"]:
                misses.append((r["id"], f"brain {dec.brain} != {r['gold_brain']}", text))
    cost = (
        COSTS["missed_medicine"] * (c["medicine"] - c["medicine_refused"])
        + COSTS["false_refusal"] * c["false_refusal"]
        + COSTS["false_silence"] * c["false_silence"]
        + COSTS["false_wake"] * c["false_wake"]
        + COSTS["ask"] * (c["ask_for_me"] + c["ask_not_for_me"])
    )
    return {"counts": dict(c), "misses": misses, "cost": cost}


def sweep(table: list[dict[str, str]], raw: dict[str, dict[str, Any]], gates) -> dict[str, float]:
    """Thresholds with the lowest error cost on the given (tuning) rows.

    Ties go to the lower medicine threshold (safer), then to the values closest to the start.
    """
    candidates = []
    for y in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        for n in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            if n > y:
                continue
            for m in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
                thr = {"addressed_yes": y, "addressed_no": n, "medicine_tau": m}
                cost = score_rows(
                    table, {k: decide(*answers_of(v), **thr) for k, v in raw.items()}, gates
                )["cost"]
                candidates.append(((cost, m, round(abs(y - 0.85) + abs(n - 0.15), 3)), thr))
    return min(candidates, key=lambda c: c[0])[1]


def frame_rows_from(path: Path | None) -> dict[str, dict[str, Any]]:
    """Phrase id -> the photo it uses (eval/frame_map.csv) plus that photo's gate and Request 2."""
    shots = {}
    if path is not None:
        for x in read_jsonl(path):
            shots[(x["label"], x["gold_frame"])] = x
    out = {}
    for m in csv.DictReader(FRAME_MAP.open()):
        level = m["level"]
        key = (m["label"], level) if m["label"] != "none" else ("none", level)
        shot = shots.get(key, {})
        out[m["id"]] = {**m, "gate": shot.get("gate"), "file": shot.get("file"), "shot": shot}
    return out


def cmd_score(args: argparse.Namespace) -> None:
    raw_all = read_jsonl(Path(args.raw) if args.raw else latest("decider-asr-*.jsonl"))
    frames_path = (
        Path(args.frames)
        if args.frames
        else next(iter(sorted((MAIN / "eval" / "out").glob("frames-*.jsonl"))[-1:]), None)
    )
    fr = frame_rows_from(frames_path)
    gates = {k: gate_status(fr, {"id": k}) for k in fr}
    table = rows()
    heard = {x["id"]: x["asr_text"] for x in raw_all}
    for r in table:
        if r["id"] in heard:
            r["_heard"] = heard[r["id"]]
    tune = [r for r in table if r["split"] == "tune"]
    held = [r for r in table if r["split"] == "held_out"]
    report: list[str] = [
        f"# Scores {stamp()} (raw {args.raw or 'latest'}; frames {frames_path.name if frames_path else 'none'})",
        "",
    ]
    results: list[dict[str, Any]] = []
    chosen: dict[str, Any] = {}

    def add(variant: str, split: str, s: dict[str, Any]) -> None:
        c = s["counts"]

        def ratio(a: str, b: str) -> str:
            return f"{c.get(a, 0)}/{c.get(b, 0)}"

        metrics = {
            "medicine_recall": ratio("medicine_refused", "medicine"),
            "medicine_recall_no_keyword": ratio("medicine_refused_nokw", "medicine_nokw"),
            "false_refusals": ratio("false_refusal", "not_medicine"),
            "false_refusals_G": f"{c.get('false_refusal_G', 0)}",
            "false_silence": ratio("false_silence", "for_me"),
            "false_wake": ratio("false_wake", "not_for_me"),
            "did_you_ask": f"{c.get('ask_for_me', 0) + c.get('ask_not_for_me', 0)}/{c.get('n', 0)}",
            "confusable_pairs": ratio("pairs_ok", "pairs_n"),
            "brain_accuracy": ratio("brain_ok", "brain_scored"),
            "no_picture_stopped": ratio("h_hold_up", "h_n"),
            "error_cost": str(s["cost"]),
        }
        for k, v in metrics.items():
            results.append({"variant": variant, "split": split, "metric": k, "value": v})
        report.append(f"## {variant} · {split}")
        report.append("| metric | value |\n|---|---|")
        report.extend(f"| {k} | {v} |" for k, v in metrics.items())
        if s["misses"]:
            report.append("\nMisses:")
            report.extend(f'- {i} · {what} · "{text}"' for i, what, text in s["misses"])
        report.append("")

    for variant in ("text", "near_mic"):
        raw = {x["id"]: x for x in raw_all if x["variant"] == variant}
        if not raw:
            continue
        thr = sweep(tune, raw, gates)
        chosen[variant] = thr
        report.append(
            f"Thresholds for `{variant}` from error costs on tune ({len(tune)} rows): `{thr}`\n"
        )
        for split, part in (("held_out", held), ("tune", tune)):
            add(
                f"router[{variant}]",
                split,
                score_rows(part, {k: decide(*answers_of(v), **thr) for k, v in raw.items()}, gates),
            )
        add(
            f"d_no_gates[{variant}]",
            "held_out",
            score_rows(
                held,
                {k: baselines.no_gates_decide(*answers_of(v)) for k, v in raw.items()},
                gates,
                use_gate=False,
            ),
        )
        # brain coverage at >= 0.9
        ok = [
            x
            for k, x in raw.items()
            if any(
                r["id"] == k
                and r["split"] == "held_out"
                and r["gold_addressed"] == "Y"
                and r["gold_medicine"] == "N"
                for r in table
            )
        ]
        gold = {r["id"]: r["gold_brain"] for r in table}
        sure = [x for x in ok if x["brain_conf"] >= 0.9]
        results.append(
            {
                "variant": f"router[{variant}]",
                "split": "held_out",
                "metric": "brain_coverage_at_0.9",
                "value": f"{len(sure)}/{len(ok)} covered, {sum(gold[x['id']] == x['brain_top'] for x in sure)}/{len(sure)} right",
            }
        )
        # tau curve
        curve = []
        for tau in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
            s = score_rows(
                held,
                {k: decide(*answers_of(v), **{**thr, "medicine_tau": tau}) for k, v in raw.items()},
                gates,
            )["counts"]
            curve.append(
                f"| {tau} | {s.get('medicine_refused', 0)}/{s.get('medicine', 0)} | {s.get('medicine_refused_nokw', 0)}/{s.get('medicine_nokw', 0)} | {s.get('false_refusal', 0)}/{s.get('not_medicine', 0)} |"
            )
        report.append(
            f"### τ_med curve, held-out, `{variant}`\n| τ_med | medicine refused | no keyword | false refusals |\n|---|---|---|---|"
        )
        report.extend(curve)
        report.append("")
    kw = {r["id"]: baselines.keyword_decide(r.get("_heard", r["script_text"])) for r in table}
    add("c_keywords", "held_out", score_rows(held, kw, gates))
    ts = stamp()
    (OUT / f"thresholds-{ts}.json").write_text(json.dumps(chosen, indent=1))
    with (ROOT / "eval" / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "split", "metric", "value"])
        w.writeheader()
        w.writerows(results)
    md = OUT / f"score-{ts}.md"
    md.write_text("\n".join(report))
    print("\n".join(report))
    print("wrote", md.relative_to(ROOT), "eval/results.csv", f"eval/out/thresholds-{ts}.json")


# ------------------------------------------------------------------ e2e (Bedrock, paid)


def load_frame(fr: dict[str, Any] | None):
    if fr is None or fr.get("level") == "missing" or not fr.get("file"):
        return None
    return frames.load(FRAMES_RAW / fr["file"])


def cmd_e2e(args: argparse.Namespace) -> None:
    from n8 import agent as n8agent
    from n8.brains import bedrock_brains

    bp = n8agent.bp
    spend = cap()
    asr = {x["id"]: x for x in read_jsonl(latest("asr-*.jsonl"))}
    thr_all = json.loads(latest("thresholds-*.json").read_text())
    thresholds = thr_all.get(args.near_mic and "near_mic" or "text")
    frames_path = next(iter(sorted((MAIN / "eval" / "out").glob("frames-*.jsonl"))[-1:]), None)
    fr = frame_rows_from(frames_path)
    table = [r for r in rows() if args.split == "all" or r["split"] == args.split]
    if args.limit:
        table = table[: args.limit]
    brains = bedrock_brains(spend)
    out = []
    print(
        f"{args.variant}: {len(table)} phrases, thresholds {thresholds}, spend before ${spend.spent():.4f}"
    )

    async def go() -> None:
        for r in table:
            a = asr.get(r["id"])
            if a is None:
                continue
            frm = fr.get(r["id"])
            img = load_frame(frm)
            t = bp.Turn(
                latest=a["text"],
                near_mic=a["near_mic"] if args.near_mic else None,
                frame=img,
                has_camera=bool(r["frame_path"]),
            )
            if args.variant == "router":
                cache = bp.DecisionCache()
                agent = n8agent.build_agent(
                    brains, DeciderClient(URL, timeout=5), cache, thresholds=thresholds
                )
                reply = await n8agent.run_turn(t, agent, cache, brains)
            elif args.variant == "pro":
                jpeg = frames.for_bedrock(img) if img is not None else None
                reply = await n8agent.always(brains.pro, t, jpeg, brains)
            else:
                reply = await classifier_turn(t, brains, bp)
            row = {
                "id": r["id"],
                "variant": args.variant,
                "split": r["split"],
                "cat": r["cat"],
                "heard": a["text"],
                "asr_final_ms": a["final_after_eos_ms"],
                **reply.__dict__,
            }
            out.append(row)
            print(
                r["id"],
                reply.action,
                reply.routed or reply.model,
                reply.verdict or "",
                "|",
                (reply.said or "")[:70],
                f"| {reply.ms.get('ready_ms')} ms ${reply.usd:.5f}",
            )

    try:
        asyncio.run(go())
    finally:
        write_jsonl(OUT / f"e2e-{args.variant}-{stamp()}.jsonl", out)
        print(
            f"bedrock actual ${sum(x['usd'] for x in out):.4f}; ledger total ${spend.spent():.4f}"
        )


async def classifier_turn(t: Any, brains: Any, bp: Any) -> Any:
    """Baseline (b): Nova Micro answers Q1/Q2 with the same wording, ClassifierStrategy picks the brain.

    Same code gates as the router (image gate, OR refusal, fixed lines); no frame_check.
    """
    from pydantic import BaseModel, Field
    from strands import Agent
    from strands.models.routing import ClassifierStrategy, ModelRouter

    from n8 import agent as n8agent

    qs = load_questions(QUESTIONS)

    class Q12(BaseModel):
        addressed_to_me: bool = Field(description=str(qs["addressed_to_me"].instructions))
        about_medicine: bool = Field(description=str(qs["about_medicine"].instructions))

    started = time.perf_counter()
    brains.take_calls()
    judge = Agent(
        model=brains.micro,
        callback_handler=None,
        system_prompt="Answer about the latest utterance spoken in a kitchen.",
    )
    res = await judge.invoke_async(f"Latest utterance: {t.latest!r}", structured_output_model=Q12)
    q: Q12 = res.structured_output
    decided = time.perf_counter()
    reply = n8agent.Reply(action="answer", said=None)
    if not q.addressed_to_me:
        reply.action = "silent"
    elif q.about_medicine:
        reply.action, reply.said = "refuse", rules.MEDICINE_REFUSE
    else:
        extra = brains.extra.get("classifier")
        if extra is None:
            from strands.models.bedrock import BedrockModel

            from n8.brains import CappedModel

            extra = CappedModel(
                BedrockModel(
                    model_id="amazon.nova-micro-v1:0",
                    region_name="us-east-1",
                    max_tokens=60,
                    temperature=0.0,
                ),
                "micro",
                brains.micro.cap,
                60,
            )
            brains.extra["classifier"] = extra
        routed: dict[str, str] = {}

        class Spy(ClassifierStrategy):
            async def select(self, context, **kw):
                c = await super().select(context, **kw)
                routed["name"] = c.name if c is not None else "declined"
                return c

        router = ModelRouter(
            bp.brain_candidates(brains.micro, brains.lite, brains.pro), strategy=Spy(extra)
        )
        content: list[dict[str, Any]] = [{"text": t.latest}]
        gate = frames.check(t.frame) if t.has_camera else None
        if gate is not None and not gate.ok:
            reply.said, reply.verdict = rules.HOLD_UP, f"gate:{gate.status}"
        else:
            if t.frame is not None:
                content.append(
                    {"image": {"format": "jpeg", "source": {"bytes": frames.for_bedrock(t.frame)}}}
                )
            agent = Agent(model=router, system_prompt=n8agent.SYSTEM_PROMPT, callback_handler=None)
            result = await agent.invoke_async([{"role": "user", "content": content}])
            reply.answer = reply.said = n8agent._text(result.message)
            reply.routed = routed.get("name")
            reply.brain = reply.routed
    calls = brains.take_calls()
    reply.calls = [n8agent._call_row(c) for c in calls]
    reply.usd = round(sum(c.usd for c in calls), 7)
    first = [c.first_token_ms for c in calls[1:] if c.first_token_ms is not None]
    reply.ms = {
        "decided_ms": round((decided - started) * 1000, 1),
        "first_token_ms": first[-1] if first else None,
        "ready_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return reply


# ------------------------------------------------------------------ bench


def path_of(x: dict[str, Any]) -> str:
    if x["action"] in ("silent", "ask", "refuse", "error"):
        return x["action"]
    if (x.get("verdict") or "").startswith("gate:"):
        return "image gate"
    return x.get("routed") or x.get("model") or "answer"


def cmd_bench(args: argparse.Namespace) -> None:
    polly = speak.Polly(cap(), cache_dir=TTS_CACHE)
    bench = ROOT / "bench"
    bench.mkdir(exist_ok=True)
    summary: dict[str, Any] = {"measured_at": stamp(), "variants": {}}
    md = [
        f"# Time to first word and cost ({stamp()})",
        "",
        "Clock: from the end of speech in the recording. Steps: Transcribe final → Decider Request 1 "
        "(earcon) → Bedrock answer ∥ Decider Request 2 → Polly first byte. Mac mini Decider over an SSH "
        "tunnel; mini hardware is not in $.",
        "",
    ]
    for variant in args.variants:
        path = latest(f"e2e-{variant}-*.jsonl")
        items = read_jsonl(path)
        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        for x in items:
            said = x.get("said")
            fb = None
            if said:
                s = polly.say(said)
                fb = s.first_byte_ms if not s.cached else None
                x["polly_first_byte_ms"] = fb
            asr = x.get("asr_final_ms") or 0.0
            ms = x.get("ms") or {}
            row = {
                "transcribe_final_ms": asr,
                "decider1_ms": ms.get("text_decider_ms"),
                "earcon_ms": asr + ms["decided_ms"]
                if ms.get("decided_ms") is not None and x["action"] != "silent"
                else None,
                "bedrock_first_token_ms": ms.get("first_token_ms"),
                "decider2_ms": ms.get("frame_decider_ms"),
                "ready_ms": asr + ms["ready_ms"] if ms.get("ready_ms") is not None else None,
                "polly_first_byte_ms": fb,
            }
            row["first_word_ms"] = (
                row["ready_ms"] + fb if (row["ready_ms"] is not None and fb is not None) else None
            )
            groups[path_of(x)].append(row)
            groups["all"].append(row)
        table = {}
        md.append(f"## {variant} ({path.name}, n={len(items)})")
        md.append(
            "| path | n | transcribe | Decider R1 | first sound | Bedrock 1st token | Decider R2 | answer ready | Polly 1st byte | **first word** p50 / p90 |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for g, rs in sorted(groups.items()):
            cols = {k: (pct([r[k] for r in rs], 0.5), pct([r[k] for r in rs], 0.9)) for k in rs[0]}
            table[g] = {"n": len(rs), **{k: {"p50": v[0], "p90": v[1]} for k, v in cols.items()}}
            def f(k: str, cols: dict[str, tuple[Any, Any]] = cols) -> str:
                return f"{cols[k][0]} / {cols[k][1]}" if cols[k][0] is not None else "–"

            md.append(
                f"| {g} | {len(rs)} | {f('transcribe_final_ms')} | {f('decider1_ms')} | {f('earcon_ms')} | "
                f"{f('bedrock_first_token_ms')} | {f('decider2_ms')} | {f('ready_ms')} | {f('polly_first_byte_ms')} | **{f('first_word_ms')}** |"
            )
        usd_bedrock = sum(x["usd"] for x in items)
        n = max(1, len(items))
        audio_s = sum(
            a["audio_s"]
            for a in read_jsonl(latest("asr-*.jsonl"))
            if a["id"] in {x["id"] for x in items}
        )
        chars = sum(len(x.get("said") or "") for x in items)
        cost = {
            "bedrock_per_1000": round(usd_bedrock / n * 1000, 4),
            "transcribe_per_1000_list": round(audio_s * 0.0001667 / n * 1000, 4),
            "polly_per_1000": round(chars * 16 / 1e6 / n * 1000, 4),
        }
        cost["full_stack_per_1000"] = round(sum(cost.values()), 4)
        table["cost"] = cost
        md.append(
            f"\n$ per 1,000 phrases: Bedrock {cost['bedrock_per_1000']} · Transcribe (list $0.0001667/s, no minimum) "
            f"{cost['transcribe_per_1000_list']} · Polly neural ($16/1M chars) {cost['polly_per_1000']} · "
            f"**full stack {cost['full_stack_per_1000']}** (Lambda not built). Prices: eval/cost.md.\n"
        )
        summary["variants"][variant] = table
    (bench / "latency.json").write_text(json.dumps(summary, indent=1))
    (bench / "latency.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"spend ${cap().spent():.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tv")
    t.add_argument("--force", action="store_true")
    a = sub.add_parser("asr")
    a.add_argument("--concurrency", type=int, default=4)
    sub.add_parser("decider")
    s = sub.add_parser("score")
    s.add_argument("--raw")
    s.add_argument("--frames")
    e = sub.add_parser("e2e")
    e.add_argument("--variant", choices=["router", "pro", "classifier"], required=True)
    e.add_argument("--split", default="all", choices=["all", "tune", "held_out"])
    e.add_argument("--near-mic", action="store_true")
    e.add_argument("--limit", type=int)
    b = sub.add_parser("bench")
    b.add_argument("--variants", nargs="+", default=["router", "pro"])
    args = p.parse_args()
    {
        "tv": cmd_tv,
        "asr": cmd_asr,
        "decider": cmd_decider,
        "score": cmd_score,
        "e2e": cmd_e2e,
        "bench": cmd_bench,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
