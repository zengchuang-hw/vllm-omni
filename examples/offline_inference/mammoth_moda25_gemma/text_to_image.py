# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline text-to-image with a trained Mammoth Gemma 3 checkpoint."""

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.model_extras.mammoth_moda25 import build_text_to_image_prompt
from vllm_omni.transformers_utils.configs.mammoth_moda25_gemma import MammothModa25GemmaConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="mammoth_gemma.png")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/mammoth_moda25_gemma.yaml")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    if not isinstance(config, MammothModa25GemmaConfig):
        raise ValueError("--model must be a trained Mammoth Gemma checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = build_text_to_image_prompt(
        tokenizer,
        config,
        f"You are a helpful image generator.\n{args.prompt}",
        args.height,
        args.width,
        system_prompt=None,
    )
    max_tokens = (args.height // 16) * (args.width // 16 + 1) + 1
    params = [
        SamplingParams(max_tokens=max_tokens, ignore_eos=True, detokenize=False, seed=args.seed),
        SamplingParams(
            max_tokens=1,
            seed=args.seed,
            extra_args={
                "text_guidance_scale": 9.0,
                "cfg_range": [0.0, 1.0],
                "num_inference_steps": 50,
            },
        ),
    ]
    omni = Omni(model=args.model, deploy_config=args.deploy_config)
    try:
        outputs = list(omni.generate(prompt, sampling_params_list=params))
        from examples.offline_inference.text_to_image.text_to_image import (
            _normalize_images_for_save,
            extract_images_from_outputs,
        )

        images = _normalize_images_for_save(extract_images_from_outputs(outputs))
        if not images:
            raise RuntimeError("No image returned by the DiT stage")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output)
        print(f"Saved {output}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
