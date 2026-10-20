"""Decider Request 2 on the photographed frames ($0): code gate first, then frame_check + item_is_medicine.

    python eval/run_frames.py eval/frames/raw    # files in shot-list order (SHOTLIST.md)
"""
import base64, io, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decider_kit.client import DeciderClient  # noqa: E402
from decider_kit.questions import load_questions  # noqa: E402
from n8 import frames  # noqa: E402

LABELS = ["soup", "pasta", "bread_mix", "rice", "soup_seasoning", "sugar", "salt", "paprika", "vitamin_d", "sleep_aid", "pain_relief"]
MEDICINE = {"vitamin_d", "sleep_aid", "pain_relief"}
SLOTS = [(lab, lvl) for lab in LABELS for lvl in ("clear", "partly", "unreadable")] + [("none", "dark"), ("none", "blurred"), ("none", "empty")]


def b64(img):
    buf = io.BytesIO(); frames.resize_for_decider(img.convert("RGB")).save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"))
    if len(files) != len(SLOTS):
        print(f"warning: {len(files)} files, shot list has {len(SLOTS)}; pairing in order")
    qs = load_questions(ROOT / "patterns/brain_picker/frame_questions.yaml")
    rows = []
    with DeciderClient(os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:18199"), timeout=20) as c:
        for f, (label, level) in zip(files, SLOTS):
            img = frames.load(f)
            g = frames.check(img)
            row = {"file": f.name, "label": label, "gold_frame": level, "gold_medicine": label in MEDICINE,
                   "gate": g.status, "mean": round(g.mean, 1), "sharpness": round(g.sharpness, 1)}
            if img is not None:
                r = c.ask("A photo from a phone worn on the chest of a cook.", qs, images=[b64(img)])
                fc, med = r.answers["frame_check"], r.answers["item_is_medicine"]
                row.update(frame=fc.choice, frame_probs=fc.probabilities, medicine=med.probability, client_ms=round(r.client_ms))
            rows.append(row)
            print(row["file"], label, level, row["gate"], row.get("frame"), round(row.get("medicine", -1), 2))
    out = ROOT / "eval/out" / f"frames-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("raw:", out.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1])
