import hashlib
import json
import os
import re
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import (
    AutoFeatureExtractor,
    WavLMForXVector,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from vllm_omni.model_executor.models.fish_speech.dac_utils import build_dac_codec

s = Path(__file__).resolve().parent
texts = json.loads((s / "scripts.json").read_text())
codecpath = Path(os.environ["DRAMA3_MODEL"]) / "codec.pth"
assert (
    hashlib.file_digest(codecpath.open("rb"), "sha256").hexdigest()
    == "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
)
with torch.inference_mode():
    c = build_dac_codec()
    state = torch.load(codecpath, map_location="cpu", weights_only=True)
    c.load_state_dict(state.get("generator", state), strict=True)
    del state
    c.eval().cuda()
    audio = {}
    for key, text in texts.items():
        codes = torch.from_numpy(np.load(s / (key + "-codes.npy"))).long().cuda()
        wav, _ = c.decode(codes.T[None], torch.tensor([len(codes)], device="cuda"))
        clip = wav.float().cpu().numpy().ravel()
        with wave.open(str(s / (key + ".wav")), "wb") as f:
            f.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            f.writeframes((clip.clip(-1, 1) * 32767).astype("<i2").tobytes())
        audio[key] = resample_poly(clip, 160, 441)
    del c
    torch.cuda.empty_cache()
    wh = Path(os.environ["WHISPER_MODEL"])
    proc = WhisperProcessor.from_pretrained(wh, local_files_only=True)
    asr = (
        WhisperForConditionalGeneration.from_pretrained(
            wh, local_files_only=True, dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    sp = Path(os.environ["WAVLM_MODEL"])
    fe = AutoFeatureExtractor.from_pretrained(sp, local_files_only=True)
    sv = WavLMForXVector.from_pretrained(sp, local_files_only=True).cuda().eval()

    def trans(a, lang):
        x = proc(a, sampling_rate=16000, return_tensors="pt")
        y = asr.generate(
            x.input_features.cuda().half(),
            language=lang,
            task="transcribe",
            num_beams=1,
            do_sample=False,
            max_new_tokens=128,
        )
        return proc.batch_decode(y, skip_special_tokens=True)[0].strip()

    def embed(a):
        x = fe(a, sampling_rate=16000, return_tensors="pt")
        v = sv(x.input_values.cuda()).embeddings[0].cpu().numpy()
        return v / np.linalg.norm(v)

    result = []
    vecs = []
    for k, txt in texts.items():
        parts = [u.strip() for u in re.split(r"\[[^\]]+\]", txt) if u.strip()]
        a = audio[k]
        lang = "english" if k.startswith("en") else "chinese"
        size = 320
        energy = np.sqrt((a[: len(a) // size * size].reshape(-1, size) ** 2).mean(1))
        active = np.flatnonzero(energy >= np.percentile(energy, 95) * 10 ** (-35 / 20))
        center = active[0] + (active[-1] - active[0]) * len(parts[0]) / (
            len(parts[0]) + len(parts[1])
        )
        dist = np.abs(np.arange(len(energy)) - center)
        # Locate a quiet 100ms interval near the expected transition.
        allowed = dist < max(12, 0.18 * (active[-1] - active[0]))
        sm = np.convolve(energy, np.ones(5) / 5, "same")
        split = (
            int(
                np.argmin(
                    np.where(allowed, sm + 0.002 * dist / max(1, dist.max()), np.inf)
                )
            )
            * size
        )
        left, right = a[:split], a[split:]
        v, w = embed(left), embed(right)
        vecs.append((k, v, w))
        row = dict(
            key=k,
            script=txt,
            hyp=trans(a, lang),
            before=trans(left, lang),
            after=trans(right, lang),
            duration=round(len(a) / 16000, 3),
            boundary=round(split / 16000, 3),
            tail_s=round((len(energy) - active[-1] - 1) * 0.02, 3),
            cosine=round(float(v @ w), 4),
        )
        result.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for row, (k, v, w) in zip(result, vecs):
        row["other_take_same_language_cosines"] = [
            round(float(v @ other_w), 4)
            for other, _, other_w in vecs
            if other[:2] == k[:2] and other != k
        ]
    (s / "audio-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
