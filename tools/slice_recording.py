"""Cut a recording into per-phrase wavs ($0, all local).

    python tools/slice_recording.py ~/Downloads/take.m4a [noise_db=-35] [min_pause=0.8]

Speech is split on pauses (ffmpeg silencedetect). Each piece is transcribed locally
(parakeet-mlx) only to line it up with the script. The script is the four recording
sections in order (1 cook · 2 cook to someone else · 3 child · 4 another adult), so a
take may cover any part of it, stop early or jump ahead. A phrase may span up to three
pieces ("Helper," … pause … "is this one still good?"). The local ASR sometimes writes
accented English in Cyrillic, so each phrase is also compared with its Russian-letter hint.
Of several takes of one phrase, the one the ASR understood best wins (on a tie, the later). Output: eval/audio/<id>.wav (16 kHz
mono) and eval/audio/slice-<take>.tsv (matched, unused pieces), plus how many are missing.
"""
import csv, difflib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval/recording"))
from pronounce_ru import RU  # noqa: E402

ROLES = {1: "cook", 2: "cook to someone else", 3: "child (or adult reading)", 4: "another adult"}
FFMPEG = "/opt/homebrew/bin/ffmpeg"
PARAKEET = Path.home() / "llm-engines/parakeet-mlx/venv/bin/parakeet-mlx"
PAD, MIN_SPEECH, FLAG_BELOW, KEEP, RETAKE, MAX_JOIN = 0.15, 0.35, 0.85, 0.7, 0.7, 3


def norm(s):
    s = s.lower().replace("́", "").replace("´", "").replace("’", "").replace("'", "").replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^a-zа-я0-9 ]", " ", s)).strip()


def sim(row, text):
    t = norm(text)
    return max(difflib.SequenceMatcher(None, norm(row["script_text"]), t).ratio(),
               difflib.SequenceMatcher(None, norm(RU.get(row["id"], "")), t).ratio() if row["id"] in RU else 0.0)


def take_score(row, text):
    """For picking among takes: a wrong word must cost more than an accent, so Latin ASR is compared by words."""
    t = norm(text)
    if re.search(r"[a-z]", t):
        return difflib.SequenceMatcher(None, norm(row["script_text"]).split(), t.split()).ratio()
    return sim(row, text)


def speech_spans(wav, noise_db, min_pause):
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", wav, "-af", f"silencedetect=noise={noise_db}dB:d={min_pause}",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if dur is None:
        raise SystemExit("ffmpeg did not report a duration")
    total = 3600 * int(dur.group(1)) + 60 * int(dur.group(2)) + float(dur.group(3))
    spans, cur = [], 0.0
    for s, e in zip(starts, ends + [total] * (len(starts) - len(ends))):
        if s - cur >= MIN_SPEECH:
            spans.append((max(0.0, cur - PAD), s + PAD))
        cur = e
    if total - cur >= MIN_SPEECH:
        spans.append((max(0.0, cur - PAD), total))
    return spans


def transcribe(paths, out_dir):
    subprocess.run([str(PARAKEET), *map(str, paths), "--output-format", "txt", "--output-dir", str(out_dir)],
                   capture_output=True, env={**os.environ, "HF_HOME": str(Path.home() / "llm-engines/hf-home")})
    outs = [Path(out_dir) / (Path(p).stem + ".txt") for p in paths]
    return [o.read_text().strip() if o.exists() else "" for o in outs]


def align(rows, heard):
    """Monotone; phrases and pieces may be skipped; a phrase takes 1..MAX_JOIN consecutive pieces.
    A match earns sim - KEEP, so only good matches count. Returns [(phrase index, first piece, last piece)]."""
    E, S = len(rows), len(heard)
    joined = lambda a, b: " ".join(heard[a:b + 1])  # noqa: E731
    best = [[0.0] * (S + 1) for _ in range(E + 1)]
    move = [[None] * (S + 1) for _ in range(E + 1)]
    for i in range(E + 1):
        for j in range(S + 1):
            if i == 0 and j == 0:
                continue
            opts = []
            if j:
                opts.append((best[i][j - 1], 0, ("s",)))
            if i:
                opts.append((best[i - 1][j], 0, ("e",)))
            for k in range(1, min(MAX_JOIN, j) + 1 if i else 1):
                opts.append((best[i - 1][j - k] + sim(rows[i - 1], joined(j - k, j - 1)) - KEEP, 1, ("m", k)))
            best[i][j], _, move[i][j] = max(opts, key=lambda o: (o[0], o[1]))
    pairs, i, j = [], E, S
    while i or j:
        m = move[i][j]
        if m[0] == "m":
            pairs.append((i - 1, j - m[1], j - 1)); i -= 1; j -= m[1]
        elif m[0] == "s":
            j -= 1
        else:
            i -= 1
    pairs = pairs[::-1]
    # Several takes of one phrase (Nadya TG 1078: "pick the best pronunciation"): among the good
    # takes up to the next matched phrase, the one the ASR understood best wins; on a tie, the later.
    for n, (i, a, b) in enumerate(pairs):
        limit = pairs[n + 1][1] if n + 1 < len(pairs) else S
        takes = [(a, b)] + [(x, y) for x in range(b + 1, limit) for y in range(x, min(x + MAX_JOIN, limit))
                            if sim(rows[i], joined(x, y)) >= RETAKE]
        pairs[n] = (i, *max(takes, key=lambda p: (round(take_score(rows[i], joined(*p)), 2), p[0])))
    return pairs


def script_order():
    rows = list(csv.DictReader((ROOT / "eval/phrases.csv").open()))
    return [r for role in ROLES.values() for r in rows if r["speaker_role"] == role]


def main(audio, noise_db=-35.0, min_pause=0.8):
    rows = script_order()
    out = ROOT / "eval/audio"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = f"{tmp}/all.wav"
        cut = lambda a, b, dst: subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", wav,  # noqa: E731
                                                "-ss", f"{a:.3f}", "-to", f"{b:.3f}", str(dst)], check=True)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(audio), "-ar", "16000", "-ac", "1", wav], check=True)
        spans = speech_spans(wav, noise_db, min_pause)
        pieces = [f"{tmp}/piece{k:03d}.wav" for k in range(len(spans))]
        for p, (a, b) in zip(pieces, spans):
            cut(a, b, p)
        heard = transcribe(pieces, tmp)
        pairs = align(rows, heard)
        report, used = [], set()
        for i, a, b in pairs:
            r, text = rows[i], " ".join(heard[a:b + 1])
            s = sim(r, text)
            cut(spans[a][0], spans[b][1], out / f"{r['id']}.wav")
            used.update(range(a, b + 1))
            report.append([r["id"], r["speaker_role"], f"{a}-{b}", f"{spans[a][0]:.2f}", f"{spans[b][1]:.2f}",
                           f"{s:.2f}", "CHECK" if s < FLAG_BELOW else "", r["script_text"], text])
    unused = [[k, f"{spans[k][0]:.2f}", heard[k]] for k in range(len(spans)) if k not in used]
    with (out / f"slice-{Path(audio).stem}.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "role", "pieces", "start_s", "end_s", "match", "flag", "script", "heard_local"])
        w.writerows(report)
        w.writerow([]); w.writerow(["unused piece", "start_s", "heard_local"]); w.writerows(unused)
    have = {p.stem for p in out.glob("p*.wav")}
    missing = [r["id"] for r in rows if r["id"] not in have]
    print(json.dumps({"take": Path(audio).name, "pieces": len(spans), "matched": len(pairs), "unused_pieces": len(unused),
                      "flagged": sum(1 for x in report if x[6]), "matched_ids": [rows[i]["id"] for i, _, _ in pairs],
                      "still_missing": len(missing)}))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), *(float(x) for x in a[1:3]))
