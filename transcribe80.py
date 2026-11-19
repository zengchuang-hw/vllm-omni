import json
import os
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging

logging.set_verbosity_error()
study = Path(__file__).resolve().parent
arm = sys.argv[1]
folder = study / "runs" / arm
texts = json.loads((folder / "metadata.json").read_text())["texts"]
ckpt = Path(os.environ["WHISPER_MODEL"])
processor = WhisperProcessor.from_pretrained(ckpt, local_files_only=True)
model = (
    WhisperForConditionalGeneration.from_pretrained(
        ckpt, local_files_only=True, dtype=torch.float16
    )
    .cuda()
    .eval()
)


def words(t):
    return processor.tokenizer.normalize(re.sub(r"\[[^\]]*\]", " ", t)).split()


def edits(a, b):
    last = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(last[j] + 1, cur[-1] + 1, last[j - 1] + (x != y)))
        last = cur
    return last[-1]


segments = []
files = sorted((folder / "wav").glob("*.wav"))
assert len(files) == 80
for f in files:
    with wave.open(str(f), "rb") as wf:
        a = (
            np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(
                np.float32
            )
            / 32768
        )
    frame = 2205
    power = np.sqrt(
        np.mean(a[: a.size // frame * frame].reshape(-1, frame) ** 2, axis=1)
    )
    loud = np.flatnonzero(
        power >= max(np.percentile(power, 95) * 10 ** (-35 / 20), 1e-10)
    )
    if len(loud):
        a = a[: min(len(a), (loud[-1] + 12) * frame)]
    a = resample_poly(a, 160, 441).astype("float32")
    while len(a) > 29 * 16000:
        lo, hi = 20 * 16000, 28 * 16000
        win = 800
        region = a[lo:hi].reshape(-1, win)
        cut = lo + win * int(np.mean(region**2, axis=1).argmin()) + win // 2
        segments.append((f.stem, a[:cut]))
        a = a[cut:]
    segments.append((f.stem, a))
joined = {}
with torch.inference_mode():
    for offset in range(0, len(segments), 8):
        batch = segments[offset : offset + 8]
        feat = processor.feature_extractor(
            [a for _, a in batch],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        ids = model.generate(
            feat.input_features.to("cuda", torch.float16),
            attention_mask=feat.attention_mask.cuda(),
            language="english",
            task="transcribe",
            num_beams=1,
            max_new_tokens=192,
            temperature=0.0,
            do_sample=False,
        )
        for (stem, _), t in zip(
            batch, processor.batch_decode(ids, skip_special_tokens=True), strict=True
        ):
            joined[stem] = joined.get(stem, "") + " " + t.strip()
rows = []
for stem, t in joined.items():
    p, s = stem.split("-")
    ref = words(texts[int(p[1:])])
    hyp = words(t)
    rows.append(
        dict(
            prompt=int(p[1:]),
            seed=int(s[1:]),
            ref=" ".join(ref),
            hyp=t.strip(),
            words=len(ref),
            errors=edits(ref, hyp),
        )
    )
(study / f"asr-{arm}.json").write_text(json.dumps(rows, indent=2))
print(arm, sum(r["errors"] for r in rows), sum(r["words"] for r in rows))
print(
    [r for r in rows if (r["prompt"], r["seed"]) in {(0, 6), (0, 16), (0, 18), (3, 7)}]
)
