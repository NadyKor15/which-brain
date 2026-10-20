# First Decider-only run, 2026-10-20 23:01Z ($0, mini, Decider 2B v1 MPS)

Input: **script text** of the 120 phrases, `{"latest": text, "previous": []}`, no loudness, wording variant A. Not the reported number: real numbers come from Transcribe text on held-out after recording. Raw: `decider-script-20261020T230127Z.jsonl`. Client latency p50 328 ms, p90 337 ms (through an SSH forward).

- **Who it was said to is the hard part.** `addressed_to_me` P(yes): to the helper median 0.71 (0.38–0.88), not to the helper median 0.59 (0.15–0.81). With 0.85 / 0.15, 81 of 90 helper phrases and 29 of 30 others land on "Did you ask me?". No false wake, no false silence, but it asks nearly always. Next: 3–5 wording variants of Q1 on the tune split, the previous two lines in the state, the loudness feature (`near_mic`), then thresholds from error costs.
- **Medicine (OR rule: P ≥ τ or brain top = ask_human), all 120, exploratory:** τ 0.30 → 27/30 refused, 4/60 false refusals; τ 0.15 → 28/30, 8/60; τ 0.10 → 30/30, 12/60. `about_medicine` alone: medicine median 0.29, other median 0.07. Brain top = ask_human on 24 of 30 medicine phrases. τ is set on the tune split only.
- **Brain (top choice, non-medicine helper phrases):** 44/60.
