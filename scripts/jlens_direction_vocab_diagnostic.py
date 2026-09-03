#!/usr/bin/env python3
"""Which direction-vocabulary tokens actually appear in the lens top-20, and do they help?

A class is only as good as the words that fire for it. This counts, per class, which of its
tokens reach the top 20 at all, and splits each token's appearances by whether that class is
the trajectory's answer -- so a discourse word riding along in every trajectory shows up as a
token with a huge count and a lift near zero.
"""

import collections
import csv
import json
import os

CL = ["UP", "DOWN", "LEFT", "RIGHT"]
vocab = json.load(open("/workspace/jlens/direction_tokens_full.json"))
tok2cls = {t: c for c in CL for t in vocab[c]}
root = os.environ.get("LENS_ROOT", "/workspace/activations/heldout360_l15")

hits = {c: collections.Counter() for c in CL}  # token -> times in a top-20
hits_when_answer = {c: collections.Counter() for c in CL}
rows_per_answer = collections.Counter()
n_rows = 0
for size in sorted(os.listdir(root)):
    d = os.path.join(root, size)
    for name in os.listdir(d):
        p = os.path.join(d, name, f"{name}_jlens_analysis.csv")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["layer"] != "15":
                    continue
                n_rows += 1
                ans = r["agent_action"]
                rows_per_answer[ans] += 1
                for i in range(1, 21):
                    t = r.get(f"top_{i}")
                    c = tok2cls.get(t)
                    if c is None:
                        continue
                    hits[c][t] += 1
                    if c == ans:
                        hits_when_answer[c][t] += 1

print(f"{n_rows} rows; answer marginal " + str({k: round(v / n_rows, 3) for k, v in rows_per_answer.most_common()}))
for c in CL:
    used = len(hits[c])
    tot = sum(hits[c].values())
    print(f"\n=== {c}: {used}/{len(vocab[c])} vocabulary tokens ever seen, {tot} hits")
    base = rows_per_answer[c] / n_rows
    print(f"    {'token':>16} {'hits':>7} {'%of class':>10} {'P(ans=c|hit)':>13}  base={base:.3f}")
    for t, n in hits[c].most_common(10):
        p = hits_when_answer[c][t] / n
        print(f"    {t!r:>16} {n:>7} {100 * n / tot:>9.1f}% {p:>13.3f}  {'+' if p > base else '-'}")
