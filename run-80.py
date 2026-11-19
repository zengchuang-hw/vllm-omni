import asyncio
import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer
from vllm.sampling_params import RequestOutputKind
from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.model_executor.models.fish_omni.fish_omni_model import (
    FishOmniForConditionalGeneration as Fish,
)
from vllm_omni.model_executor.models.fish_omni.pipeline import FISH_OMNI_PIPELINE

study = Path(__file__).resolve().parent
outdir = study / "cohort80"
model = os.environ["DRAMA3_MODEL"]
pipeline = FISH_OMNI_PIPELINE
first = replace(
    pipeline.stages[0],
    final_output=True,
    final_output_type="text",
    async_chunk_process_next_stage_input_func=None,
)
register_pipeline(replace(pipeline, stages=(first,)))
texts = [
    "[sighs] Fine. If we miss it, it is on you. [laughs] Well, maybe that is fair.",
    "Good evening. The lights are on, the room is quiet, and we are ready to begin.",
    "The rain has stopped, and the street is quiet. I left a note on the kitchen table. Please read it before you leave, then call me when you reach the station.",
    "[softly] I thought you had forgotten. [sighs] I am glad you came back.",
]


async def main():
    outdir.mkdir(exist_ok=True)
    config = yaml.safe_load((study / "review-stage0.yaml").read_text())
    stage = config["stages"][0]
    stage.update(max_num_seqs=8)
    stage["compilation_config"].update(
        max_cudagraph_capture_size=8, cudagraph_capture_sizes=[1, 2, 4, 8]
    )
    configfile = outdir / "stage0.yaml"
    configfile.write_text(yaml.safe_dump(config))
    tokenizer = AutoTokenizer.from_pretrained(model)
    eng = AsyncOmni(model=model, deploy_config=str(configfile), stage_init_timeout=900)
    limit = asyncio.Semaphore(8)

    async def generate(index, seed):
        async with limit:
            key = f"p{index}-s{seed:02d}"
            info = {"text": texts[index]}
            length = Fish.estimate_prompt_len_from_additional_information(
                info, tokenizer=tokenizer
            )
            prompt = {"prompt_token_ids": [1] * length, "additional_information": info}
            params = copy.deepcopy(eng.default_sampling_params_list)
            params[0].max_tokens = 1000
            params[0].seed = seed
            params[0].output_kind = RequestOutputKind.FINAL_ONLY
            final = None
            async for output in eng.generate(
                prompt, request_id=key, sampling_params_list=params
            ):
                if output.finished:
                    final = output
            if final is None:
                raise RuntimeError(f"missing finished output for {key}")
            mm = final.multimodal_output
            codes = mm.get("codes.audio")
            if codes is None:
                codes = mm["codes"]["audio"]
            if isinstance(codes, list):
                codes = torch.cat(codes)
            frames = torch.as_tensor(codes).reshape(-1, 11).cpu()
            acoustic = frames[
                (frames[:, 0] >= 248099) & (frames[:, 0] < 248099 + 4096), 1:
            ]
            assert len(acoustic) > 0
            np.save(outdir / (key + ".npy"), acoustic.numpy())
            reason = final.outputs[0].finish_reason
            assert reason in ("stop", "length"), (key, reason)
            print(key, len(acoustic), reason, flush=True)
            return dict(
                key=key,
                prompt=index,
                seed=seed,
                text=texts[index],
                reason=reason,
                frames=len(acoustic),
            )

    try:
        rows = await asyncio.gather(
            *(generate(i, seed) for seed in range(20) for i in range(len(texts)))
        )
        (outdir / "codes.json").write_text(json.dumps(rows, indent=2))
    finally:
        eng.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
