import json
import os
from pathlib import Path

from tests.helpers.speaker_similarity import load_audio_16k
from vllm import LLM, SamplingParams

s = Path(__file__).resolve().parent


def main():
    model = LLM(
        model=os.environ["QWEN_ASR_MODEL"],
        max_model_len=2048,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        limit_mm_per_prompt={"audio": 1},
        max_num_seqs=4,
    )
    prompt = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    result = {}
    for r in json.loads((s / "audio-results.json").read_text()):
        k = r["key"]
        a = load_audio_16k(s / (k + ".wav"))
        cut = round(r["boundary"] * 16000)
        req = [
            {"prompt": prompt, "multi_modal_data": {"audio": (wave, 16000)}}
            for wave in (a, a[cut:])
        ]
        out = model.generate(
            req, SamplingParams(temperature=0, max_tokens=100), use_tqdm=False
        )
        result[k] = {
            name: o.outputs[0].text.strip()
            for name, o in zip(("full", "last_clause"), out, strict=True)
        }
        print(k, result[k], flush=True)
    (s / "independent-asr.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
