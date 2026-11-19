import hashlib
import json
import os
from pathlib import Path

from tests.helpers.speaker_similarity import (
    CampPlusEmbedder,
    WavLMSVEmbedder,
    load_audio_16k,
)

s = Path(__file__).resolve().parent
rows = json.loads((s / "audio-results.json").read_text())
camp = Path(os.environ["CAMP_MODEL"])
wavlm = Path(os.environ["WAVLM_MODEL"])
result = {
    "campplus_sha256": hashlib.file_digest(
        (camp / "campplus.onnx").open("rb"), "sha256"
    ).hexdigest(),
    "results": {},
}
for enc in (CampPlusEmbedder(camp), WavLMSVEmbedder(str(wavlm))):
    left = {}
    right = {}
    for r in rows:
        k = r["key"]
        wav = load_audio_16k(s / (k + ".wav"))
        cut = round(r["boundary"] * 16000)
        left[k] = enc.embed(wav[:cut])
        right[k] = enc.embed(wav[cut:])
    result["results"][enc.name] = {
        i: {j: round(float(v @ right[j]), 4) for j in right if i[:2] == j[:2]}
        for i, v in left.items()
    }
(s / "speakers.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
