# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 3 AR -> Mammoth DiT/VAE."""

from dataclasses import replace

from vllm_omni.model_executor.models.mammoth_moda2.pipeline import MAMMOTH_MODA2_PIPELINE

MAMMOTH_MODA25_GEMMA_PIPELINE = replace(
    MAMMOTH_MODA2_PIPELINE,
    model_type="mammoth_moda25_gemma",
    model_arch="MammothModa25GemmaForConditionalGeneration",
    hf_architectures=("MammothModa25GemmaForConditionalGeneration",),
    default_deploy_config_name="mammoth_moda25_gemma.yaml",
    stages=(
        replace(MAMMOTH_MODA2_PIPELINE.stages[0], model_arch="MammothModa25GemmaForConditionalGeneration"),
        replace(MAMMOTH_MODA2_PIPELINE.stages[1], model_arch="MammothModa2DiTPipeline"),
    ),
)
