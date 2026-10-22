# brain_picker — Strands Decider 2B picks the Bedrock brain, safety first

**What it decides, per spoken phrase, before any model is called** (`questions.yaml`):

| Question | Type | Plain-code rule (`src/n8/decide.py`) |
|---|---|---|
| `addressed_to_me` | noul | P(yes) ≥ yes → go on · ≤ no → silence · between → "Did you ask me?" once |
| `about_medicine` | noul | P(yes) ≥ τ_med → refuse (τ well below 0.5) |
| `brain` | choice: `quiet_answer` / `look` / `read_closely` / `ask_human` | conf ≥ 0.90 → that brain; below → the stronger of the top two; `ask_human` on top → refuse |

Refuse = `about_medicine` **or** `brain == ask_human` **or** a model answer that names a medicine or a dose
(`src/n8/agent.py: answer_guard`). `item_is_medicine` on the frame is asked too, but it is **logged only**:
on 2026-10-22 it false-alarmed on 9 of 36 food photos at P(yes) ≥ 0.30, above the brief's 10 % bar
(`ITEM_MEDICINE_ACTS = False` in `src/n8/decide.py`; `docs/RESULTS-2026-10-22.md`). The refusal
is a fixed line in code: "I won't guess on medicine. Want me to call your helper?" The call screen is
simulated and runs only after a "yes".

Thresholds on a noul are set on **P(yes)**, never on "confidence": Decider returns P(true) only, and
`max(p, 1 - p)` hides which side it is on. Values come from error costs on the tuning split
(`eval/run_live.py score`, costs in the brief's task card: missed medicine 100, false silence 5,
false wake 2, false refusal 1, "Did you ask me?" 1).

## Drop-in (Strands 1.61.1, `strands.models.routing` is provisional)

```python
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.routing import ModelRouter
from decider_kit import DeciderClient
from handler import DecisionCache, DeciderRoutingStrategy, SafetyFirstHandler, Turn, TURN_KEY, brain_candidates

micro, lite, pro = (BedrockModel(model_id=f"amazon.nova-{s}-v1:0") for s in ("micro", "lite", "pro"))
cache = DecisionCache()
agent = Agent(
    model=ModelRouter(brain_candidates(micro, lite, pro), strategy=DeciderRoutingStrategy(cache)),
    interventions=[SafetyFirstHandler(DeciderClient("http://127.0.0.1:8099"), cache)],
)
await agent.invoke_async("What's in this can?", invocation_state={TURN_KEY: Turn("What's in this can?", frame=img)})
```

The one-line swap from the SDK's own LLM router (baseline b in our eval):

```python
ModelRouter(brain_candidates(micro, lite, pro), strategy=ClassifierStrategy(BedrockModel(model_id="amazon.nova-micro-v1:0")))
ModelRouter(brain_candidates(micro, lite, pro), strategy=DeciderRoutingStrategy(cache))
```

## How it is wired

- `SafetyFirstHandler.before_invocation` asks Decider Request 1. Silence, "Did you ask me?", the
  refusal and "Hold it up to the camera." are `Deny`: code speaks, no model runs. For a look or a close
  read it runs the code image gate, starts Request 2 on the frame **next to** the Bedrock call, and
  attaches the frame (`Transform`). `on_error="deny"`: a handler exception cancels the turn and code
  says "Sorry, something went wrong. Please ask your helper."
- The decision reaches the router two ways: `invocation_state[DECISION_KEY]` (the live dict Strands
  passes to `RoutingContext`) and `DecisionCache` by agent identity, for call paths where Strands
  builds a fresh `invocation_state`. Both are unit-tested.
- `DeciderRoutingStrategy.select` returns the candidate **instance** from `context.candidates` (the
  router matches by identity). After a failed call it asks `FallbackStrategy` over the candidates
  above the one that failed, so it only escalates up.
- Decider unreachable or slower than 1.5 s: the keyword net (`MEDICINE_KEYWORDS`) refuses on any hit
  and everything else goes to the strongest brain, logged `degraded=true`.
- Request 2 down, slower than 2.5 s or erroring: nothing about the frame is said ("Sorry, something
  went wrong. Please ask your helper."), because the medicine check on the frame did not run.
- Last checks in code on every model answer (`src/n8/agent.py: speak_reply`): medicine words or a
  dose in the answer → the refusal; "I think" never next to a number; with a usable frame in view,
  `item_is_medicine` is asked for every brain and logged (it does not refuse; see above).

## Measured

Numbers live in `eval/results.csv` (held-out split) and `bench/latency.md`; this README quotes only
what those files contain.

## Tests

`tests/test_brain_picker.py` — fake Decider server (no model load) and the real Strands `Agent` +
`ModelRouter` with scripted models: OR refusal, P(yes) thresholds and the ask band, ladder escalation,
candidate by identity on both decision paths, fail-closed on server down / timeout / handler
exception, image gate before Bedrock, frame verdicts, the spend cap before every model call.
