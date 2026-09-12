# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan-OCR backbone with Mammoth's generation vocabulary and FFN expert."""

from torch import nn
from vllm.distributed import get_pp_group
from vllm.model_executor.models.hunyuan_v1 import HunYuanDecoderLayer, HunYuanMLP, _get_cla_factor, _is_moe
from vllm.model_executor.models.hunyuan_vision import (
    HunYuanVisionTransformer,
    HunYuanVLDummyInputsBuilder,
    HunYuanVLForConditionalGeneration,
    HunYuanVLMultiModalProcessor,
    HunYuanVLProcessingInfo,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2ARForConditionalGeneration,
    MammothModa2Qwen2ForCausalLM,
    moe_forward,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


class MammothModa25DecoderLayer(HunYuanDecoderLayer):
    def __init__(self, config, layer_idx, cache_config=None, quant_config=None, prefix=""):
        if _get_cla_factor(config) != 1 or _is_moe(config):
            raise ValueError("MammothModa2.5 requires the dense Hunyuan-OCR backbone without cross-layer attention")
        super().__init__(config, cache_config, quant_config, prefix, layer_id=layer_idx)
        self.gen_mlp = (
            HunYuanMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.gen_mlp",
            )
            if config.moe_type == "ffn"
            else None
        )

    def forward(self, positions, hidden_states, residual, gen_token_mask=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, _ = self.self_attn(positions=positions, hidden_states=hidden_states, kv_states=None)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return moe_forward(hidden_states, self.mlp, self.gen_mlp, gen_token_mask), residual


class MammothModa25ForCausalLM(MammothModa2Qwen2ForCausalLM):
    """Reuse vocabulary/PP plumbing, replacing every decoder with Hunyuan."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=MammothModa25DecoderLayer)
        if self.config.tie_word_embeddings and get_pp_group().is_last_rank:
            self.lm_head = self.lm_head.tie_weights(self.embed_tokens)


class MammothModa25ProcessingInfo(HunYuanVLProcessingInfo):
    def get_hf_config(self):
        config = self.ctx.get_hf_config()
        return getattr(config, "llm_config", config)


@MULTIMODAL_REGISTRY.register_processor(
    HunYuanVLMultiModalProcessor,
    info=MammothModa25ProcessingInfo,
    dummy_inputs=HunYuanVLDummyInputsBuilder,
)
class MammothModa25ForConditionalGeneration(HunYuanVLForConditionalGeneration):
    have_multimodal_outputs = True
    requires_raw_input_tokens = True

    # Proposed 2.5 checkpoint layout: llm_model contains the OCR backbone
    # plus gen_embed_tokens/gen_head/gen_mlp; DiT/VAE retain Mammoth2 names.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "gen_image_condition_refiner.": None,
            "gen_transformer.": None,
            "gen_vae.": None,
            "llm_model.vit.vit.": "visual.",
            "llm_model.vit.": "visual.",
            "llm_model.model.": "language_model.",
            "llm_model.lm_head.": "language_model.lm_head.",
            "llm_model.gen_head.": "language_model.gen_head.",
        }
    )

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        if vllm_config.lora_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("MammothModa2.5 does not yet support LoRA or speculative decoding")
        config = vllm_config.model_config.hf_config
        self.config = getattr(config, "llm_config", config)
        if self.config is None or self.config.model_type != "mammothmoda25_hunyuan_vl":
            raise ValueError("Use a MammothModa25 checkpoint with an extended Hunyuan-OCR llm_config")
        ar_config = vllm_config.with_hf_config(self.config)
        with self._mark_tower_model(ar_config, {"image"}):
            self.visual = HunYuanVisionTransformer(
                self.config.vision_config,
                quant_config=ar_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(ar_config):
            self.language_model = MammothModa25ForCausalLM(
                vllm_config=ar_config, prefix=maybe_prefix(prefix, "language_model")
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self._last_runtime_additional_information = None

    _apply_t2i_token_constraints = MammothModa2ARForConditionalGeneration._apply_t2i_token_constraints

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        runtime = kwargs.get("runtime_additional_information")
        self._last_runtime_additional_information = runtime if isinstance(runtime, list) else None
        if input_ids is None and intermediate_tensors is None and self.config.moe_type == "ffn":
            raise ValueError("MammothModa2.5 requires input_ids to route generation tokens to their FFN expert")
        hidden = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(hidden, IntermediateTensors):
            return hidden
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

    def compute_logits(self, hidden_states):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = self.language_model.compute_logits(hidden_states)
        return self._apply_t2i_token_constraints(logits)

    def load_weights(self, weights):
        loaded = AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
        required = {name for name, _ in self.named_parameters() if ".gen_" in name}
        missing = required - loaded
        if missing:
            raise ValueError(f"MammothModa2.5 checkpoint is missing trained generation weights: {sorted(missing)}")
        return loaded
