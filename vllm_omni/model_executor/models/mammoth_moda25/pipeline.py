# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan-OCR AR -> MammothModa2 DiT/VAE."""

from dataclasses import replace

from vllm_omni.model_executor.models.mammoth_moda2.pipeline import MAMMOTH_MODA2_PIPELINE

MAMMOTH_MODA25_PIPELINE = replace(
    MAMMOTH_MODA2_PIPELINE,
    model_type="mammoth_moda25",
    model_arch="MammothModa25ForConditionalGeneration",
    hf_architectures=("MammothModa25ForConditionalGeneration",),
    default_deploy_config_name="mammoth_moda25.yaml",
    stages=(
        replace(MAMMOTH_MODA2_PIPELINE.stages[0], model_arch="MammothModa25ForConditionalGeneration"),
        replace(MAMMOTH_MODA2_PIPELINE.stages[1], model_arch="MammothModa2DiTPipeline"),
    ),
)
