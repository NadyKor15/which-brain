"""Live check: does the installed Strands ClassifierStrategy route on Amazon Nova, or decline?

    python tools/check_classifier_fix.py [N]     # N text phrases (default 6), through the spend cap

Uses the SDK's own ClassifierStrategy with Nova Micro as the classifier and the three brains as
candidates, exactly as baseline (b) but without our workaround. Prints the Strands version and,
per phrase, the candidate the strategy picked or "declined". Run it once on a release (expected:
all declined, issue #5431) and once on a fix (expected: real picks).
Output: eval/out/check-classifier-fix-<version>-<UTC>.json
"""
import asyncio, csv, json, sys, time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from strands import Agent  # noqa: E402
from strands.models import ModelRouter  # noqa: E402
from strands.models.bedrock import BedrockModel  # noqa: E402
from strands.models.routing import ClassifierStrategy  # noqa: E402

from n8.agent import SYSTEM_PROMPT  # noqa: E402
from n8.brains import CappedModel, bedrock_brains  # noqa: E402
from n8.spend import SpendCap  # noqa: E402
from patterns.brain_picker import handler as bp  # noqa: E402


def phrases(n: int) -> list[dict]:
    rows = list(csv.DictReader(open(ROOT / "eval/phrases.csv")))
    picked, brains_seen = [], set()
    for r in rows:  # one per gold brain first, then fill
        if r["gold_addressed"] == "Y" and not r["frame_path"] and r["gold_brain"] not in brains_seen:
            picked.append(r)
            brains_seen.add(r["gold_brain"])
    picked += [r for r in rows if r["gold_addressed"] == "Y" and not r["frame_path"] and r not in picked]
    return picked[:n]


async def main(n: int) -> None:
    cap = SpendCap(ROOT / "eval/out/spend.jsonl")
    brains = bedrock_brains(cap)
    classifier = CappedModel(
        BedrockModel(model_id="amazon.nova-micro-v1:0", region_name="us-east-1", max_tokens=200, temperature=0.0),
        "micro", cap, 200,
    )
    out = {"strands_agents": version("strands-agents"), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    for r in phrases(n):
        strategy = ClassifierStrategy(classifier)
        inner, routed = strategy.select, {}

        async def spy(context, _inner=inner, _routed=routed, **kw):
            c = await _inner(context, **kw)
            _routed["name"] = c.name if c is not None else "declined"
            return c

        strategy.select = spy
        router = ModelRouter(bp.brain_candidates(brains.micro, brains.lite, brains.pro), strategy=strategy)
        err = None
        try:  # the routing choice is recorded before the answer; a long answer is not the point
            await Agent(model=router, system_prompt=SYSTEM_PROMPT, callback_handler=None).invoke_async(r["script_text"])
        except Exception as exc:
            err = type(exc).__name__
        out["rows"].append({"id": r["id"], "gold_brain": r["gold_brain"], "routed": routed.get("name"), "answer_error": err})
        print(r["id"], r["gold_brain"], "->", routed.get("name"), err or "")
    path = ROOT / f"eval/out/check-classifier-fix-{out['strands_agents']}-{out['ts'].replace(':', '').replace('-', '')}.json"
    path.write_text(json.dumps(out, indent=1))
    print(out["strands_agents"], "declined", sum(x["routed"] == "declined" for x in out["rows"]), "of", len(out["rows"]), "->", path.name)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
