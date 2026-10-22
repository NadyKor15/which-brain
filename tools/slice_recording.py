"""Cut one long section recording into per-phrase wavs ($0, all local).

    python tools/slice_recording.py 1 ~/Downloads/section1.m4a   # 1 cook · 2 cook to someone else · 3 child · 4 another adult

Speech is split on pauses (ffmpeg silencedetect). Each piece is transcribed locally
(parakeet-mlx) only to line it up with the script: a monotone alignment matches every
script phrase to one piece and skips the rest (false starts, repeats after a mistake;
on a repeat the later take wins). Output: eval/audio/<id>.wav (16 kHz mono) and
eval/audio/slice-section<N>.tsv for an eye check. Pieces with low text match are flagged.
"""
import csv, difflib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = {1: "cook", 2: "cook to someone else", 3: "child (or adult reading)", 4: "another adult"}
FFMPEG = "/opt/homebrew/bin/ffmpeg"
PARAKEET = Path.home() / "llm-engines/parakeet-mlx/venv/bin/parakeet-mlx"
PAD, MIN_SPEECH, FLAG_BELOW = 0.15, 0.35, 0.5


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("’", "'").replace("'", ""))


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


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
    return [(Path(out_dir) / (Path(p).stem + ".txt")).read_text().strip()
            if (Path(out_dir) / (Path(p).stem + ".txt")).exists() else "" for p in paths]


def align(expected, heard):
    """Monotone: each expected phrase gets exactly one piece, pieces may be skipped. Max total similarity."""
    E, S = len(expected), len(heard)
    if S < E:
        raise SystemExit(f"only {S} speech pieces for {E} phrases — lower --noise or --pause, or re-record")
    neg = float("-inf")
    best = [[neg] * (S + 1) for _ in range(E + 1)]
    take = [[False] * (S + 1) for _ in range(E + 1)]
    for j in range(S + 1):
        best[0][j] = 0.0
    for i in range(1, E + 1):
        for j in range(i, S + 1):
            use = best[i - 1][j - 1] + sim(expected[i - 1], heard[j - 1])
            skip = best[i][j - 1]
            take[i][j] = use >= skip  # tie → this (later) piece wins
            best[i][j] = max(use, skip)
    pairs, i, j = [], E, S
    while i > 0:
        if take[i][j]:
            pairs.append((i - 1, j - 1)); i -= 1
        j -= 1
    return pairs[::-1]


def main(section, audio, noise_db=-35.0, min_pause=0.8):
    rows = [r for r in csv.DictReader((ROOT / "eval/phrases.csv").open()) if r["speaker_role"] == ROLES[section]]
    out = ROOT / "eval/audio"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = f"{tmp}/all.wav"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(audio), "-ar", "16000", "-ac", "1", wav], check=True)
        spans = speech_spans(wav, noise_db, min_pause)
        pieces = []
        for k, (a, b) in enumerate(spans):
            p = f"{tmp}/piece{k:03d}.wav"
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", wav, "-ss", f"{a:.3f}", "-to", f"{b:.3f}", p], check=True)
            pieces.append(p)
        heard = transcribe(pieces, tmp)
        pairs = align([r["script_text"] for r in rows], heard)
        report, flagged = [], 0
        for i, j in pairs:
            r, s = rows[i], sim(rows[i]["script_text"], heard[j])
            Path(pieces[j]).replace(out / f"{r['id']}.wav")
            flagged += s < FLAG_BELOW
            report.append([r["id"], j, f"{spans[j][0]:.2f}", f"{spans[j][1]:.2f}", f"{s:.2f}",
                           "CHECK" if s < FLAG_BELOW else "", r["script_text"], heard[j]])
    with (out / f"slice-section{section}.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "piece", "start_s", "end_s", "match", "flag", "script", "heard_local"])
        w.writerows(report)
    print(json.dumps({"section": section, "phrases": len(rows), "pieces": len(spans),
                      "skipped_pieces": len(spans) - len(pairs), "flagged": flagged}))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), Path(a[1]), *(float(x) for x in a[2:4]))
