"""Real system output for the video overlay: chosen recorded phrases, end to end.

    python eval/demo_turns.py p056 p118 p096 p001 p074 --near-mic=code   # paid, capped

For each phrase: what Transcribe heard, Decider's numbers (Q1-Q3 and, for a look, frame_check and
item_is_medicine), the brain badge, the words said and the timing strip. Writes
eval/out/demo-<stamp>/overlay.json (data for the editor; no staged values) and the reply and earcon
audio to the main checkout's eval/audio/demo-<stamp>/ (not in git).
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import run_live as rl  # noqa: E402

from decider_kit import DeciderClient  # noqa: E402
from n8 import agent as n8agent  # noqa: E402
from n8 import audio, earcons, speak  # noqa: E402
from n8.brains import bedrock_brains  # noqa: E402


def main(ids: list[str], near_mic: str) -> None:
    bp = n8agent.bp
    stamp = rl.stamp()
    out_dir = rl.OUT / f"demo-{stamp}"
    wav_dir = rl.AUDIO / f"demo-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    earcons.write_all(wav_dir / "earcons")
    spend = rl.cap()
    brains = bedrock_brains(spend)
    polly = speak.Polly(spend)
    table = {r["id"]: r for r in csv.DictReader(rl.PHRASES.open())}
    thr = json.loads(rl.latest("thresholds-*.json").read_text()).get(rl.THRESHOLD_KEY[near_mic])
    fr = rl.frame_rows_from(
        next(iter(sorted((rl.MAIN / "eval" / "out").glob("frames-*.jsonl"))[-1:]), None)
    )
    cache_asr = audio.TranscriptCache(rl.ASR_CACHE)
    turns = []

    async def go() -> None:
        for i in ids:
            r = table[i]
            t = await audio.transcribe_file(i, rl.AUDIO / f"{i}.wav", cap=spend, cache=cache_asr)
            asr_row = {x["id"]: x for x in rl.read_jsonl(rl.latest("asr-*.jsonl"))}.get(i, {})
            img = rl.load_frame(fr.get(i))
            turn = bp.Turn(
                latest=t.text,
                near_mic=asr_row.get("near_mic"),
                frame=img,
                has_camera=bool(r["frame_path"]),
            )
            cache = bp.DecisionCache()
            agent = n8agent.build_agent(
                brains, DeciderClient(rl.URL, timeout=5), cache, thresholds=thr, near_mic=near_mic
            )
            reply = await n8agent.run_turn(turn, agent, cache, brains)
            said = reply.said
            fb = None
            if said:
                s = polly.say(said)
                fb = s.first_byte_ms
                speak.write_wav(wav_dir / f"{i}-reply.wav", s.pcm)
            turns.append(
                {
                    "id": i,
                    "script": r["script_text"],
                    "heard": t.text,
                    "speaker_role": r["speaker_role"],
                    "near_mic": turn.near_mic,
                    "decider": reply.probs,
                    "frame": reply.frame_probs,
                    "action": reply.action,
                    "brain_badge": reply.routed,
                    "model": reply.model,
                    "verdict": reply.verdict,
                    "earcon": reply.earcon,
                    "model_answer": reply.answer,
                    "said": said,
                    "timing_ms": {
                        "transcribe_final_after_speech": t.final_after_eos_ms,
                        **reply.ms,
                        "polly_first_byte": fb,
                    },
                    "usd": reply.usd,
                    "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            print(i, reply.action, reply.routed or "", reply.verdict or "", "|", said)

    asyncio.run(go())
    (out_dir / "overlay.json").write_text(json.dumps(turns, indent=1, default=str))
    print("wrote", (out_dir / "overlay.json").relative_to(ROOT), "audio:", wav_dir)
    print(f"ledger ${spend.spent():.4f}")


if __name__ == "__main__":
    mode = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--near-mic=")), "off")
    main([a for a in sys.argv[1:] if not a.startswith("--")], mode)
