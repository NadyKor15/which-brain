# Which Brain?

![tests]({REPO_URL}/actions/workflows/tests.yml/badge.svg)

A hands-free kitchen helper for people who can't read a label. The phone is worn on the chest. For every phrase it hears, Strands Decider 2B decides whether the phrase was meant for the helper, whether it is about medicine, and which Amazon Nova model on Bedrock should answer: a quick one, one that looks at the label, or one that reads closely. On medicine it never guesses: it offers to call a person.

Built for the AWS Builder Center "Agent Experience Hackathon: Strands Decider" (#strands-decider-2b, #model-routing). Article: {ARTICLE_URL} · Video: {VIDEO_URL}

## Check it in 90 seconds

Held-out set = 84 of 120 recorded phrases; thresholds were set on the other 36. Every number below is in [`docs/RESULTS-2026-10-22.md`](docs/RESULTS-2026-10-22.md) with the file it comes from.

| Claim | Evidence | What you'll see |
|---|---|---|
| Never guesses on medicine | `eval/e2e_results.csv` | 21/21 medicine questions refused, 10/10 of those with no medicine word; 0 model answers spoken. Always Nova Pro: 21/21 left to the model |
| Knows when it is not being asked | `eval/out/score-20261022T145704Z.md`, `eval/e2e_results.csv` | answered 5 of 22 lines meant for someone else (always Nova Pro: 22 of 22) |
| Faster to the first word | `bench/latency.md` | 821 ms p50 from the end of speech (always Nova Pro: 1,228 ms); first sound 582 ms |
| Cheaper | `bench/latency.md` | $0.11 of Bedrock per 1,000 phrases (always Nova Pro: $0.83) |
| Never "I think" next to a number | `eval/e2e_results.csv` | 0 of 84 replies |
| What safety costs | `eval/out/e2e-score-20261022T145944Z.md` | 12 of 41 ordinary questions also refused; every one listed |
| A platform finding, reproduced | `tools/repro_classifier_nova.py` | Strands `ClassifierStrategy` declines every choice on Nova; output in `eval/out/repro-classifier-nova-20261022T152125Z.json` |
| Drop-in for the Strands model router | [`patterns/brain_picker/`](patterns/brain_picker/README.md) | a one-line swap from `ClassifierStrategy` |

## How it works

```
voice (phone on the chest)
  → Amazon Transcribe (final text ~224 ms after speech ends)
  → far voice? silent, in code, no model called
  → Strands Decider 2B, request 1: addressed_to_me · about_medicine · brain   (~340 ms)
      → not for me: silence · unsure: "Did you ask me?" · medicine / ask_human: fixed refusal in code
  → Strands Agent + ModelRouter(DeciderRoutingStrategy): Nova Micro · Nova Lite (+photo) · Nova Pro (+photo)
      next to it, request 2 on the photo: frame_check · item_is_medicine (logged only)
  → last checks in code on the answer (medicine word or dose → refusal; no "I think" next to a number)
  → Amazon Polly (first byte ~65 ms)
```

- `patterns/brain_picker/`: `SafetyFirstHandler` (a Strands intervention: silence, "Did you ask me?" and the refusal are spoken by code, no model runs) and `DeciderRoutingStrategy` for `ModelRouter`.
- `patterns/say_how_sure/`: the camera part on its own: a code image gate, Decider's `frame_check`, then how sure the helper may sound.
- `src/n8/`: rules (`decide.py`), Transcribe and Polly (`audio.py`, `speak.py`), the three Nova brains with a spend cap (`brains.py`, `spend.py`), baselines (`baselines.py`).
- Decider runs as a local server (`STRANDS_DECIDER_URL`, default `http://127.0.0.1:18199`); in our runs it was on a Mac mini.

## Rules the helper keeps (each one has a test)

- It never says "I think" next to a number, a date, a weight or a temperature.
- Medicine: it never states a dose. It says "I won't guess on medicine. Want me to call your helper?" and dials only after you say yes (the call screen is simulated).
- No action without "yes".
- A sound comes first: two notes for "hold still, I'm looking".
- The answer models are told to read only text they can see; there is no OCR check at run time.

## Platform finding: Strands `ClassifierStrategy` declines every choice on Amazon Nova

`ClassifierStrategy` (strands-agents 1.61.1) asks for its choice through a tool named after its pydantic class, `_ClassifierSelection`. Nova on Bedrock answers with a tool named `ClassifierSelection`, without the underscore, even when the tool is forced; `BedrockModel.structured_output` finds no tool use with the expected name, raises, and the strategy declines, so the router always serves candidate zero. Repro: `python tools/repro_classifier_nova.py` (four Nova Micro calls). Workaround: `src/n8/baselines.py`, `nova_classifier_strategy`. Upstream issue: {ISSUE_URL}.

## Run it

```bash
git clone {REPO_URL} && cd which-brain
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                                            # 163 tests, no network, no AWS, no Decider server

# Re-score from the saved Decider and end-to-end outputs ($0, no network)
python eval/run_live.py score --dry
python eval/run_live.py e2e-score --variants router pro classifier
```

The scripts find their inputs through `git`, so run them from a clone, not from a zip download. The full pipeline (`eval/run_live.py tv | asr | decider | e2e | bench`) needs a running Strands Decider 2B server, AWS credentials with Bedrock, Transcribe and Polly in us-east-1, and the recorded audio, which is not in the repo (it is the author's voice). Paid calls go through `src/n8/spend.py`, which prices every call before it is made and stops at $5; the whole build spent $1.56.

## What this does not show

- Not tested with blind or low-vision users. One survey participant's written answers shaped the rules.
- Scripted, recorded phrases from one speaker, who also plays the child (1.5 m) and the other adult (3 m); the radio and TV lines are Amazon Polly. No real child is in the recording.
- Not a pill identifier and not medical advice.
- Decider runs on a Mac mini over an SSH tunnel; the AWS Lambda host was not built.
- One run per variant, no repeats; latency depends on the mini's load at the time.

The full list is in `docs/RESULTS-2026-10-22.md`.

## How AI was used

An AI coding assistant wrote the code, tests and documents under my direction. I recorded the phrases, took the photos, chose what the helper must never do, and checked the claims against the measurements.

## Credits

`src/decider_kit/` comes from RetailBox Automation's Decider toolkit (MIT, Copyright (c) 2026 Mikhail Oskola, see `src/decider_kit/LICENSE`). Notes on Strands interventions: `docs/STRANDS-INTERVENTIONS-NOTES.md`.

## License

MIT, see `LICENSE`.
