"""Minimal repro: Strands ClassifierStrategy declines every choice on Amazon Nova (strands-agents 1.61.1).

    python tools/repro_classifier_nova.py          # 4 Nova Micro calls, < $0.001, through the spend cap

1. Raw Bedrock Converse, tool forced by name: "_ClassifierSelection" vs "ClassifierSelection".
   Prints the tool name Nova returns in toolUse.
2. The SDK path: BedrockModel.structured_output with the SDK's own pydantic model
   (strands.models.routing.classifier_strategy._ClassifierSelection) vs the same schema renamed
   without the underscore. Prints what comes back.
Output: eval/out/repro-classifier-nova-<UTC>.json
"""
import asyncio, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import boto3  # noqa: E402
from strands.models.bedrock import BedrockModel  # noqa: E402
from strands.models.routing import classifier_strategy as cs  # noqa: E402
from n8.spend import SpendCap, nova_usd  # noqa: E402

MODEL, REGION = "amazon.nova-micro-v1:0", "us-east-1"
CAP = SpendCap(ROOT / "eval/out/spend.jsonl")
PROMPT = "Pick candidate 1 for this request: 'What's in this can?'"
SCHEMA = cs._ClassifierSelection.model_json_schema()


def raw(name):
    CAP.charge(nova_usd("micro", 1500, 300), f"repro classifier raw {name}")
    r = boto3.client("bedrock-runtime", region_name=REGION).converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": PROMPT}]}],
        toolConfig={"tools": [{"toolSpec": {"name": name, "description": "Return the routing choice.",
                                            "inputSchema": {"json": SCHEMA}}}],
                    "toolChoice": {"tool": {"name": name}}},
        inferenceConfig={"maxTokens": 200, "temperature": 0},
    )
    uses = [c["toolUse"] for c in r["output"]["message"]["content"] if "toolUse" in c]
    return {"tool_asked": name, "stop_reason": r.get("stopReason"),
            "tool_returned": [u["name"] for u in uses], "input": [u.get("input") for u in uses]}


async def sdk(model_cls):
    CAP.charge(nova_usd("micro", 1500, 300), f"repro classifier sdk {model_cls.__name__}")
    m = BedrockModel(model_id=MODEL, region_name=REGION, max_tokens=200, temperature=0)
    out, err = None, None
    try:
        async for ev in m.structured_output(model_cls, [{"role": "user", "content": [{"text": PROMPT}]}]):
            if isinstance(ev, dict) and "output" in ev:
                out = ev["output"]
    except Exception as e:  # the SDK raises when no tool use with the expected name comes back
        err = f"{type(e).__name__}: {e}"
    return {"pydantic_class": model_cls.__name__, "output": repr(out), "error": err}


def main():
    renamed = type("ClassifierSelection", (cs._ClassifierSelection,), {})
    res = {"strands_agents": __import__("importlib.metadata").metadata.version("strands-agents"),
           "model": MODEL, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "raw": [raw("_ClassifierSelection"), raw("ClassifierSelection")],
           "sdk": [asyncio.run(sdk(cs._ClassifierSelection)), asyncio.run(sdk(renamed))]}
    out = ROOT / f"eval/out/repro-classifier-nova-{res['ts'].replace(':', '').replace('-', '')}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print("saved:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
