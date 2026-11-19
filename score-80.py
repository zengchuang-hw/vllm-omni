import hashlib
import itertools
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from vllm_omni.model_executor.models.fish_speech.dac_utils import build_dac_codec

s = Path(__file__).resolve().parent
cohort = s / "cohort80"
runs = s / "runs/final-80"
wavdir = runs / "wav"
wavdir.mkdir(parents=True, exist_ok=True)
rows = json.loads((cohort / "codes.json").read_text())
assert len({(r["prompt"], r["seed"]) for r in rows}) == 80
texts = {r["prompt"]: r["text"] for r in rows}
(runs / "metadata.json").write_text(json.dumps({"texts": [texts[i] for i in range(4)]}))
weights = Path(os.environ["DRAMA3_MODEL"]) / "codec.pth"
with weights.open("rb") as f:
    assert (
        hashlib.file_digest(f, "sha256").hexdigest()
        == "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
    )
with torch.inference_mode():
    codec = build_dac_codec()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    codec.load_state_dict(state.get("generator", state), strict=True)
    del state
    codec.eval().cuda()
    for r in rows:
        key = r["key"]
        codes = torch.from_numpy(np.load(cohort / (key + ".npy"))).long().cuda()
        clip, _ = codec.decode(codes.T[None], torch.tensor([len(codes)], device="cuda"))
        pcm = (clip.float().cpu().numpy().ravel().clip(-1, 1) * 32767).astype("<i2")
        with wave.open(str(wavdir / (key + ".wav")), "wb") as f:
            f.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            f.writeframes(pcm.tobytes())
        a = pcm.astype(np.float64)
        frame = 2205
        power = np.sqrt(
            np.mean(a[: len(a) // frame * frame].reshape(-1, frame) ** 2, axis=1)
        )
        quiet = power < max(np.percentile(power, 95) * 10 ** (-35 / 20), 1e-10)
        lengths = [(bool(k), sum(1 for _ in g)) for k, g in itertools.groupby(quiet)]
        r["longest_quiet_s"] = max((n for k, n in lengths if k), default=0) / 20
        r["trailing_quiet_s"] = lengths[-1][1] / 20 if lengths[-1][0] else 0
        r["duration_s"] = round(len(a) / 44100, 4)
    del codec
    torch.cuda.empty_cache()
(cohort / "scores.json").write_text(json.dumps(rows, indent=2))
print(
    "cap",
    sum(r["reason"] == "length" for r in rows),
    "long_quiet",
    sum(r["longest_quiet_s"] >= 10 for r in rows),
    "longest_tail",
    max(r["trailing_quiet_s"] for r in rows),
    flush=True,
)
subprocess.run([sys.executable, str(s / "transcribe80.py"), "final-80"], check=True)
