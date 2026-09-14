# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 3 attention, normalization and vision with Mammoth generation experts."""

import torch
from torch import nn
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.gemma3 import Gemma3DecoderLayer, Gemma3MLP
from vllm.model_executor.models.gemma3_mm import (
    Gemma3DummyInputsBuilder,
    Gemma3ForConditionalGeneration,
    Gemma3MultiModalProcessor,
    Gemma3MultiModalProjector,
    Gemma3ProcessingInfo,
)
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2ARForConditionalGeneration,
    MammothModa2Qwen2ForCausalLM,
    moe_forward,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


class MammothGemma3DecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config, cache_config=None, quant_config=None, prefix=""):
        super().__init__(config, cache_config, quant_config, prefix)
        self.gen_mlp = (
            Gemma3MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_activation,
                quant_config=quant_config,
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
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
        hidden_states = moe_forward(hidden_states, self.mlp, self.gen_mlp, gen_token_mask)
        return self.post_feedforward_layernorm(hidden_states), residual


class MammothGemma3ForCausalLM(MammothModa2Qwen2ForCausalLM):
    """Share Mammoth's split vocabulary and PP forward; retain Gemma numerics."""

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.extra_gen_vocab = True
        self.gen_vocab_start_index = self.base_vocab_size = config.gen_vocab_start_index
        self.gen_vocab_size = config.gen_vocab_size
        first, last = get_pp_group().is_first_rank, get_pp_group().is_last_rank
        for name, size in (("embed_tokens", self.base_vocab_size), ("gen_embed_tokens", self.gen_vocab_size)):
            module = (
                VocabParallelEmbedding(
                    size, config.hidden_size, quant_config=self.quant_config, prefix=maybe_prefix(prefix, name)
                )
                if first or (last and config.tie_word_embeddings)
                else PPMissingLayer()
            )
            setattr(self, name, module)
        for name, size in (("lm_head", self.base_vocab_size), ("gen_head", self.gen_vocab_size)):
            module = (
                ParallelLMHead(
                    size, config.hidden_size, quant_config=self.quant_config, prefix=maybe_prefix(prefix, name)
                )
                if last
                else PPMissingLayer()
            )
            setattr(self, name, module)
        if last and config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.embed_tokens)
        self.logits_processor = LogitsProcessor(self.base_vocab_size, soft_cap=config.final_logit_softcapping)
        self.gen_logits_processor = LogitsProcessor(self.gen_vocab_size, soft_cap=config.final_logit_softcapping)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MammothGemma3DecoderLayer(config, vllm_config.cache_config, self.quant_config, prefix),
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) if last else PPMissingLayer()
        self.register_buffer("normalizer", torch.tensor(config.hidden_size**0.5), persistent=False)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(batch_size, self.config.hidden_size, dtype=dtype, device=device),
                "residual": torch.zeros(batch_size, self.config.hidden_size, dtype=dtype, device=device),
                "gen_token_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
            }
        )

    def get_input_embeddings(self, input_ids):
        # Normalize token embeddings once, before visual embeddings are merged.
        embeddings = super().get_input_embeddings(input_ids)
        return embeddings * self.normalizer.to(dtype=embeddings.dtype)


class MammothGemma3ProcessingInfo(Gemma3ProcessingInfo):
    def get_hf_config(self):
        config = self.ctx.get_hf_config()
        return getattr(config, "llm_config", config)


@MULTIMODAL_REGISTRY.register_processor(
    Gemma3MultiModalProcessor,
    info=MammothGemma3ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class MammothModa25GemmaForConditionalGeneration(Gemma3ForConditionalGeneration):
    have_multimodal_outputs = True
    requires_raw_input_tokens = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "gen_image_condition_refiner.": None,
            "gen_transformer.": None,
            "gen_vae.": None,
            "llm_model.model.language_model.": "language_model.",
            "llm_model.model.vision_tower.": "vision_tower.",
            "llm_model.model.multi_modal_projector.": "multi_modal_projector.",
            "llm_model.lm_head.": "language_model.lm_head.",
            "llm_model.gen_head.": "language_model.gen_head.",
        }
    )

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        if vllm_config.lora_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("Mammoth Gemma does not yet support LoRA or speculative decoding")
        outer = vllm_config.model_config.hf_config
        config = getattr(outer, "llm_config", outer)
        if config is None or config.model_type != "mammothmoda25_gemma3":
            raise ValueError("Use a Mammoth Gemma checkpoint with an extended Gemma 3 llm_config")
        ar_config = vllm_config.with_hf_config(config)
        self.config = config
        self.model_config = ar_config.model_config
        self.quant_config = ar_config.quant_config
        self.multimodal_config = self.model_config.multimodal_config
        self.vit_positions_per_patch = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
        self.configure_mm_token_handling(
            vocab_size=config.text_config.vocab_size, mm_token_ids=[config.image_token_index]
        )
        with self._mark_tower_model(ar_config, "image"):
            self.vision_tower = SiglipVisionModel(
                config.vision_config, self.quant_config, prefix=maybe_prefix(prefix, "vision_tower")
            )
            self.multi_modal_projector = Gemma3MultiModalProjector(config)
        with self._mark_language_model(ar_config):
            self.language_model = MammothGemma3ForCausalLM(
                vllm_config=ar_config, prefix=maybe_prefix(prefix, "language_model")
            )
        scale = getattr(config, "logit_scale", 1.0)
        self.language_model.logits_processor.scale *= scale
        self.language_model.gen_logits_processor.scale *= scale
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self._last_runtime_additional_information = None

    _apply_t2i_token_constraints = MammothModa2ARForConditionalGeneration._apply_t2i_token_constraints

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        runtime = kwargs.get("runtime_additional_information")
        self._last_runtime_additional_information = runtime if isinstance(runtime, list) else None
        if input_ids is None and intermediate_tensors is None and self.config.moe_type == "ffn":
            raise ValueError("Gemma generation expert routing requires input_ids")
        hidden = self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(hidden, IntermediateTensors):
            return hidden
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

    def compute_logits(self, hidden_states):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self._apply_t2i_token_constraints(self.language_model.compute_logits(hidden_states))

    def load_weights(self, weights):
        loaded = AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
        missing = {name for name, _ in self.named_parameters() if ".gen_" in name} - loaded
        if missing:
            raise ValueError(f"Mammoth Gemma checkpoint is missing trained generation weights: {sorted(missing)}")
        return loaded
