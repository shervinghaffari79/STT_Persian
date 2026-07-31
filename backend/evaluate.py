#!/usr/bin/env python3
"""
Score a produced transcript against a human ground truth.

    python evaluate.py ground_truth.txt prediction.txt

Exists because every tuning decision in this pipeline was previously argued
from eyeballing a few Persian paragraphs, which cannot distinguish "the ASR
got worse" from "the ASR is fine and one segment got emitted twice". Those
have opposite fixes, and on the first file scored here they differed by 20 WER
points.

Formats:
  ground truth  alternating "Speaker N" lines and their text (what a human
                annotator naturally produces)
  prediction    this backend's export: "[S1]: ..." lines

Reported:
  WER/CER  after Persian-aware normalization (see _norm) -- the transcription
           quality proper, speaker labels ignored
  I/D/S    insertions especially: a duplicated segment shows up as a large
           insertion count while substitutions stay flat, which is how you
           tell a repeat artifact from a genuine accuracy regression
  speakers predicted vs reference speaker count

A prediction covering only part of the recording (a truncated export, or a job
still streaming) is scored against the best-matching PREFIX of the reference,
so partial output is not punished as if the missing audio were errors. The
covered fraction is always printed -- read it before trusting the WER.
"""
import re
import sys
import unicodedata

ZWNJ = "‌"

# Persian text has several ways to write the same thing, and none of them are
# transcription errors: Arabic vs Persian yeh/kaf, optional ZWNJ inside
# compounds ("می‌کنم" / "میکنم"), optional diacritics. Scoring without folding
# these first measures orthography, not recognition.
_CHAR_FOLD = [("ي", "ی"), ("ك", "ک"), ("ة", "ه"), ("أ", "ا"), ("إ", "ا"),
              ("آ", "ا"), ("ؤ", "و"), ("ئ", "ی")]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for a, b in _CHAR_FOLD:
        text = text.replace(a, b)
    text = text.replace(ZWNJ, "")
    text = re.sub(r"[ً-ْ]", "", text)   # harakat
    text = re.sub(r"[^\w\s]", " ", text)          # punctuation, both scripts
    return re.sub(r"\s+", " ", text.lower()).strip()


def _edit(ref, hyp):
    """Levenshtein distance + operation counts (ref -> hyp)."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    ops = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        ops[0][j] = j          # all insertions
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ops[i][0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def _edit_detail(ref, hyp):
    """Same distance, plus a S/D/I breakdown. Separate from _edit because the
    backtrace needs the full O(n*m) table while the prefix search below only
    needs the number, and that search runs the comparison many times."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = (d[i - 1][j - 1] if ref[i - 1] == hyp[j - 1]
                       else 1 + min(d[i - 1][j - 1], d[i - 1][j], d[i][j - 1]))
    i, j = n, m
    c = {"S": 0, "D": 0, "I": 0, "=": 0}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            c["="] += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            c["S"] += 1; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            c["D"] += 1; i -= 1
        else:
            c["I"] += 1; j -= 1
    return d[n][m], c


def parse_reference(raw: str):
    """Alternating 'Speaker N' / text lines -> (speakers, texts)."""
    speakers, texts, cur = [], [], None
    for line in (l.strip() for l in raw.splitlines()):
        if not line:
            continue
        m = re.fullmatch(r"Speaker\s+(\w+)\s*:?", line)
        if m:
            cur = f"Speaker {m.group(1)}"
            continue
        speakers.append(cur)
        texts.append(line)
    return speakers, texts


def parse_prediction(raw: str):
    """'[S1]: text' lines -> (speakers, texts)."""
    speakers, texts = [], []
    for line in raw.splitlines():
        m = re.match(r"\[(\w+)\]\s*:\s*(.*)", line.strip())
        if m and m.group(2).strip():
            speakers.append(m.group(1))
            texts.append(m.group(2).strip())
    return speakers, texts


def _best_prefix(ref_tokens, hyp_tokens):
    """Length of the reference prefix the hypothesis best corresponds to.

    Scored by error RATE rather than raw distance: raw distance always falls
    as the prefix shrinks (fewer reference words left to miss), so minimising
    it would collapse the window to nothing."""
    lo = max(1, int(len(hyp_tokens) * 0.6))
    hi = min(len(ref_tokens), max(lo, int(len(hyp_tokens) * 2.0)))
    best = (float("inf"), len(ref_tokens))
    step = max(1, (hi - lo) // 60)          # coarse sweep, then refine
    for end in range(lo, hi + 1, step):
        r = _edit(ref_tokens[:end], hyp_tokens) / end
        if r < best[0]:
            best = (r, end)
    lo2, hi2 = max(1, best[1] - step), min(len(ref_tokens), best[1] + step)
    for end in range(lo2, hi2 + 1):
        r = _edit(ref_tokens[:end], hyp_tokens) / end
        if r < best[0]:
            best = (r, end)
    return best[1]


def main(ref_path, hyp_path):
    ref_spk, ref_txt = parse_reference(open(ref_path, encoding="utf-8").read())
    hyp_spk, hyp_txt = parse_prediction(open(hyp_path, encoding="utf-8").read())
    if not hyp_txt:
        sys.exit(f"no '[S1]: ...' segments found in {hyp_path}")

    ref_words = _norm(" ".join(ref_txt)).split()
    hyp_words = _norm(" ".join(hyp_txt)).split()

    print(f"reference : {len(ref_txt):4d} utterances  {len(ref_words):5d} words  "
          f"{len(set(ref_spk))} speakers")
    print(f"prediction: {len(hyp_txt):4d} segments    {len(hyp_words):5d} words  "
          f"{len(set(hyp_spk))} speakers {sorted(set(hyp_spk))}")

    end = _best_prefix(ref_words, hyp_words)
    covered = end / len(ref_words) * 100
    dist, c = _edit_detail(ref_words[:end], hyp_words)
    print(f"\ncovered   : ~{covered:.0f}% of the reference "
          f"({end}/{len(ref_words)} words)")
    if covered < 90:
        print("            ^ partial output -- WER below is for this prefix only")
    print(f"WER       : {dist / end * 100:5.1f}%   "
          f"S={c['S']}  D={c['D']}  I={c['I']}  correct={c['=']}")
    if c["I"] > c["S"]:
        print("            ^ insertions exceed substitutions: suspect repeated/"
              "duplicated segments rather than mis-recognition")

    # Characters come from the word prefix already aligned above, NOT from a
    # separately-guessed character cut: the two windows must describe the same
    # span of speech or CER and WER end up scoring different audio.
    ref_chars = list(" ".join(ref_words[:end]))
    hyp_chars = list(" ".join(hyp_words))
    cd = _edit(ref_chars, hyp_chars)
    print(f"CER       : {cd / max(1, len(ref_chars)) * 100:5.1f}%")

    print(f"\nspeakers  : reference {len(set(ref_spk))}, predicted {len(set(hyp_spk))}", end="")
    if len(set(hyp_spk)) < len(set(ref_spk)):
        print("  -- under-clustered: try PYANNOTE_NUM_SPEAKERS / "
              "PYANNOTE_MIN_SPEAKERS, or lower PYANNOTE_THRESHOLD")
    elif len(set(hyp_spk)) > len(set(ref_spk)):
        print("  -- over-clustered: try PYANNOTE_MAX_SPEAKERS, or raise "
              "PYANNOTE_THRESHOLD")
    else:
        print("  -- matches")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
