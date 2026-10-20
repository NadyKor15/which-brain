"""Measure Q1 wording variants on the tune split ($0). D is phrased the other way round."""
import csv, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decider_kit.client import DeciderClient  # noqa: E402
from decider_kit.questions import load_questions  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "tune"
rows = [r for r in csv.DictReader((ROOT / "eval/phrases.csv").open()) if r["split"] == split]
variants = load_questions(ROOT / "patterns/brain_picker/q1_variants.yaml")
url = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:18199")
out = {}
with DeciderClient(url, timeout=10) as c:
    for name, q in variants.items():
        ps = []
        for r in rows:
            p = c.ask(json.dumps({"latest": r["script_text"], "previous": []}), {"q": q}).answers["q"].probability
            ps.append(1 - p if name == "D" else p)
        out[name] = ps
gold = [r["gold_addressed"] == "Y" for r in rows]
for name, ps in out.items():
    pos = sorted(p for p, g in zip(ps, gold) if g); neg = sorted(p for p, g in zip(ps, gold) if not g)
    auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    best = max((sum((p >= t) == g for p, g in zip(ps, gold)), t) for t in [i / 100 for i in range(5, 96)])
    print(f"{name}: AUC {auc:.3f}  med yes {pos[len(pos)//2]:.2f} no {neg[len(neg)//2]:.2f}  best acc {best[0]}/{len(ps)} at {best[1]:.2f}")
p = ROOT / "eval/out" / f"q1-variants-{split}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
p.write_text(json.dumps({"ids": [r["id"] for r in rows], "probs": out}))
print("raw:", p.relative_to(ROOT))
