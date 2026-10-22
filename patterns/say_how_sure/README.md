# say_how_sure — how sure may the helper sound about what the camera sees?

Decider without an image still answers with high confidence, so the first check is plain code.

| Step | What | Result |
|---|---|---|
| 1. `image_gate(frame)` | no frame, mean luminance below 40, or variance of the Laplacian below 60 (start values, set on the photo set) | "Hold it up to the camera." — no model is called |
| 2. `look(client, frame)` | Decider Request 2 (`questions.yaml`): `frame_check` choice clear / partly / unreadable, `item_is_medicine` noul | a verdict |
| 3. `hedge(verdict, answer)` | plain → the answer · hedge → "I think …" for words; for numbers, dates, weights and temperatures the visible start of the text plus "turn it", never "I think" next to a number · reask → "I can't read it yet. Hold it a bit closer." · refuse → the fixed medicine line | the words said |

Rules: `clear` ≥ 0.90 → plain; otherwise hedge; `unreadable` ≥ 0.60 → re-ask, **at most twice**
(`ReaskLoop`; Strands Guide loops have no retry cap), then hedge; `item_is_medicine` P(yes) ≥ 0.30 →
refuse. Decider down or slower than 2.5 s → hedge, never plain.

```python
from handler import ReaskLoop, image_gate, look, speak
loop = ReaskLoop()
gate = image_gate(frame)
result = await look(client, frame, reasks=loop.count) if gate.ok else None
say(speak(gate, result, bedrock_answer))
```

In Which Brain? the same steps run inside `patterns/brain_picker/handler.py` (Request 2 next to the
Bedrock call). This folder is the frame part on its own, for any agent that answers about a camera
frame. `questions.yaml` has the same text as `brain_picker/frame_questions.yaml`; a test keeps them
equal.

Tests: `tests/test_say_how_sure.py` (fake Decider server, no model load).
