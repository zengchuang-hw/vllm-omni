import copy
import json
import os
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer
from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.fish_omni.fish_omni_model import (
    FishOmniForConditionalGeneration as Fish,
)
from vllm_omni.model_executor.models.fish_omni.pipeline import (
    FISH_OMNI_PIPELINE as pipe,
)

s = Path(__file__).resolve().parent
model = os.environ["DRAMA3_MODEL"]
# Register a one-stage topology for the experiment. The production stage-0
# implementation, checkpoint, tokenizer and model-state code are used as shipped.
stage = replace(
    pipe.stages[0],
    final_output=True,
    final_output_type="text",
    async_chunk_process_next_stage_input_func=None,
)
register_pipeline(replace(pipe, stages=(stage,)))
texts = [
    (
        "en1",
        "[softly] The last train leaves at seven tonight. [sighs] Please wait for me beside the old clock.",
    ),
    (
        "en2",
        "[whisper] I put the blue envelope on the table. [laughs] The ticket was inside my coat all along.",
    ),
    (
        "en3",
        "[excited] We found the path before the sun came up! [softly] Follow the river until you see a stone bridge.",
    ),
    ("zh1", "[轻声] 今天下午我们去公园散步。[叹气] 记得带上雨伞，天气可能会变。"),
    ("zh2", "[sighs] 我把钥匙放在门口的桌子上。[laughs] 原来你已经找到那本书了。"),
    ("zh3", "[轻声] 火车明天早上八点出发。[笑] 我们在车站门口的钟楼下面见面。"),
]


def main():
    src = Path.cwd()
    cfg = yaml.safe_load((src / "vllm_omni/deploy/fish_omni.yaml").read_text())
    cfg["stages"] = cfg["stages"][:1]
    cfg.pop("connectors", None)
    cfg.pop("platforms", None)
    cfg["async_chunk"] = False
    stage0 = cfg["stages"][0]
    stage0.pop("output_connectors", None)
    stage0.update(max_num_seqs=4, max_model_len=4096)
    stage0["compilation_config"].update(
        max_cudagraph_capture_size=4, cudagraph_capture_sizes=[1, 2, 4]
    )
    path = s / "review-stage0.yaml"
    path.write_text(yaml.safe_dump(cfg))
    tok = AutoTokenizer.from_pretrained(model)
    (s / "scripts.json").write_text(json.dumps(dict(texts), ensure_ascii=False))
    with closing(
        Omni(model=model, deploy_config=str(path), stage_init_timeout=900)
    ) as eng:
        for key, txt in texts:
            info = {"text": txt}
            n = Fish.estimate_prompt_len_from_additional_information(
                info, tokenizer=tok
            )
            sp = copy.deepcopy(eng.default_sampling_params_list)
            sp[0].seed = 17
            sp[0].max_tokens = 1000
            out = eng.generate(
                [{"prompt_token_ids": [1] * n, "additional_information": info}],
                sampling_params_list=sp,
                use_tqdm=False,
            )[0]
            mm = out.multimodal_output
            print(
                key,
                list(mm),
                [(k, type(v).__name__, getattr(v, "shape", "")) for k, v in mm.items()],
                flush=True,
            )
            row = mm.get("codes.audio")
            if row is None:
                row = mm["codes"]["audio"]
            if isinstance(row, list):
                row = torch.cat(row)
            frames = torch.as_tensor(row).reshape(-1, 11).cpu()
            acoustic = frames[
                (frames[:, 0] >= 248099) & (frames[:, 0] < 248099 + 4096), 1:
            ]
            assert len(acoustic) > 0
            np.save(s / (key + "-codes.npy"), acoustic.numpy())
            print(key, len(acoustic), out.outputs[0].finish_reason, flush=True)


if __name__ == "__main__":
    main()
