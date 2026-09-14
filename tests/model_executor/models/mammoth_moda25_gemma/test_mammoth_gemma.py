# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig, Gemma3Config
from vllm.model_executor.models.gemma3 import Gemma3DecoderLayer

from vllm_omni.model_executor.models.mammoth_moda25_gemma import mammoth_moda25_gemma as impl
from vllm_omni.model_executor.models.mammoth_moda25_gemma.pipeline import MAMMOTH_MODA25_GEMMA_PIPELINE
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit
from vllm_omni.model_extras.mammoth_moda25 import build_text_to_image_prompt
from vllm_omni.transformers_utils.configs.mammoth_moda25_gemma import MammothModa25GemmaConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def config():
    return MammothModa25GemmaConfig(
        llm_config={
            "model_type": "gemma3",
            "gen_vocab_size": 16,
            "moe_type": "ffn",
            "text_config": {
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 16,
                "vocab_size": 64,
                "final_logit_softcapping": 10.0,
            },
            "image_token_index": 60,
            "boi_token_index": 61,
            "eoi_token_index": 62,
        },
        gen_token_config={
            "image_start_token_id": 64,
            "image_token_id": 65,
            "eol_token_id": 66,
            "visual_token_start_id": 67,
            "visual_token_end_id": 79,
        },
    )


def test_config_roundtrip_preserves_gemma(config, tmp_path):
    original = deepcopy(config.to_dict())
    config.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path)
    for cfg in (config, restored):
        assert isinstance(cfg.llm_config, Gemma3Config)
        text = cfg.get_text_config()
        assert text.vocab_size == 80
        assert text.base_vocab_size == 64
        assert text.layer_types == ["sliding_attention", "full_attention"]
        assert text.sliding_window == 16
        assert text.final_logit_softcapping == 10.0
        assert text.hidden_activation == "gelu_pytorch_tanh"
        assert cfg.image_token_id == 60
        assert cfg.vision_start_token_id == 61
        assert cfg.vision_end_token_id == 62
        assert "full_attention" in text.rope_parameters
        assert "sliding_attention" in text.rope_parameters
    assert config.to_dict() == original


@pytest.mark.parametrize("model_type", ["gemma", "gemma2", "hunyuan_vl", "gemma3_text"])
def test_reject_other_backbones(model_type):
    with pytest.raises(ValueError, match="multimodal Gemma 3"):
        MammothModa25GemmaConfig(llm_config={"model_type": model_type})


class Norm(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, hidden, residual=None):
        if residual is not None:
            residual = hidden + residual
            return residual * self.factor, residual
        return hidden * self.factor


class Attention(nn.Module):
    def forward(self, positions, hidden_states, **kwargs):
        return hidden_states * 0.5


def layer(cls):
    result = cls.__new__(cls)
    nn.Module.__init__(result)
    result.input_layernorm = Norm(2.0)
    result.post_attention_layernorm = Norm(3.0)
    result.pre_feedforward_layernorm = Norm(4.0)
    result.post_feedforward_layernorm = Norm(5.0)
    result.self_attn = Attention()
    result.mlp = nn.Identity()
    if cls is impl.MammothGemma3DecoderLayer:
        result.gen_mlp = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            result.gen_mlp.weight.copy_(2 * torch.eye(2))
    return result


@pytest.mark.parametrize("residual", [None, torch.ones(3, 2)])
def test_native_gemma_normalization_order_and_expert_routing(residual):
    native = layer(Gemma3DecoderLayer)
    adapted = layer(impl.MammothGemma3DecoderLayer)
    hidden = torch.arange(6).reshape(3, 2).float()
    positions = torch.arange(3)
    expected, expected_residual = native(positions, hidden, residual)
    output, actual_residual = adapted(positions, hidden, residual, torch.tensor([False, True, False]))
    expected[1] *= 2
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(actual_residual, expected_residual)


def bare_lm(config):
    model = impl.MammothGemma3ForCausalLM.__new__(impl.MammothGemma3ForCausalLM)
    nn.Module.__init__(model)
    model.config = config.get_text_config()
    model.quant_config = None
    model.extra_gen_vocab = True
    model.base_vocab_size = model.gen_vocab_start_index = 64
    model.gen_vocab_size = 16
    model.embed_tokens = nn.Embedding(64, 8)
    model.gen_embed_tokens = nn.Embedding(16, 8)
    model.gen_head = nn.Linear(8, 16, bias=False)
    model.register_buffer("normalizer", torch.tensor(8**0.5))
    return model


def test_token_embedding_scaling_once(config, monkeypatch):
    lm = bare_lm(config)
    with torch.no_grad():
        lm.embed_tokens.weight.fill_(1)
        lm.gen_embed_tokens.weight.fill_(2)
    actual = lm.embed_input_ids(torch.tensor([3, 67]))
    expected = torch.tensor([1.0, 2.0])[:, None].expand(2, 8) * lm.normalizer
    torch.testing.assert_close(actual, expected)
    lm.layers = nn.ModuleList([])
    lm.start_layer = lm.end_layer = 0
    lm.norm = Norm(1.0)

    # Supply residual explicitly through a fake terminal norm to examine the
    # shared forward's embedding path without running attention kernels.
    class FinalNorm(nn.Module):
        def forward(self, hidden, residual):
            return hidden, residual

    lm.norm = FinalNorm()
    import vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 as common

    monkeypatch.setattr(common, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    supplied = torch.full((2, 8), 7.0)
    torch.testing.assert_close(lm(torch.tensor([3, 67]), torch.arange(2), inputs_embeds=supplied), supplied)
    torch.testing.assert_close(lm(torch.tensor([3, 67]), torch.arange(2)), expected)


def test_gemma_weight_mapping(config):
    model = impl.MammothModa25GemmaForConditionalGeneration.__new__(impl.MammothModa25GemmaForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = bare_lm(config)
    model.vision_tower = nn.Linear(2, 2, bias=False)
    model.multi_modal_projector = nn.Linear(2, 2, bias=False)
    weights = [
        ("llm_model.model.language_model.embed_tokens.weight", torch.ones(64, 8)),
        ("llm_model.model.language_model.gen_embed_tokens.weight", torch.full((16, 8), 2.0)),
        ("llm_model.gen_head.weight", torch.full((16, 8), 3.0)),
        ("llm_model.model.vision_tower.weight", torch.ones(2, 2)),
        ("llm_model.model.multi_modal_projector.weight", torch.full((2, 2), 4.0)),
        ("gen_transformer.ignored.weight", torch.empty(1)),
    ]
    with pytest.raises(ValueError, match="missing trained generation"):
        model.load_weights(iter(weights[:1]))
    loaded = model.load_weights(iter(weights))
    assert "language_model.gen_head.weight" in loaded
    torch.testing.assert_close(model.multi_modal_projector.weight, torch.full((2, 2), 4.0))


def test_gemma_prompt_and_dit_bridge(config):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert [item["role"] for item in messages] == ["user"]
            return [2, 3]

        def encode(self, text, **kwargs):
            assert text == "2*1"
            return [4]

    prompt = build_text_to_image_prompt(Tokenizer(), config, "draw", 16, 32, system_prompt=None)
    assert prompt["prompt_token_ids"] == [2, 3, 64, 4, 65]
    source = SimpleNamespace(
        prompt_token_ids=prompt["prompt_token_ids"],
        outputs=[
            SimpleNamespace(
                cumulative_token_ids=[67, 68, 66, 67],
                multimodal_output={"latent": torch.ones(8, 8)},
            )
        ],
    )
    converted = ar2dit([source], prompt)[0]["additional_information"]
    assert converted["full_hidden_states"].shape == (8, 8)
    assert converted["full_token_ids"][-3:] == [67, 68, 66]
    ar, dit = MAMMOTH_MODA25_GEMMA_PIPELINE.stages
    assert ar.model_arch == "MammothModa25GemmaForConditionalGeneration"
    assert dit.model_arch == "MammothModa2DiTPipeline"
    assert dit.input_sources == (0,)


def test_constructor_preserves_hybrid_layers_and_tied_head(config, monkeypatch):
    import vllm.model_executor.models.gemma3 as native

    class Embedding(nn.Embedding):
        def __init__(self, size, hidden, **kwargs):
            super().__init__(size, hidden)

    class Head(nn.Linear):
        def __init__(self, size, hidden, **kwargs):
            super().__init__(hidden, size, bias=False)

        def tie_weights(self, embedding):
            self.weight = embedding.weight
            return self

    class NativeAttention(Attention):
        def __init__(self, config, prefix, **kwargs):
            super().__init__()
            self.layer_type = config.layer_types[int(prefix.split(".")[-2])]

    def mlp(hidden_size, intermediate_size, hidden_activation, **kwargs):
        assert hidden_activation == "gelu_pytorch_tanh"
        return nn.Linear(hidden_size, hidden_size, bias=False)

    monkeypatch.setattr(impl, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(impl, "VocabParallelEmbedding", Embedding)
    monkeypatch.setattr(impl, "ParallelLMHead", Head)
    monkeypatch.setattr(native, "Gemma3Attention", NativeAttention)
    monkeypatch.setattr(native, "Gemma3MLP", mlp)
    monkeypatch.setattr(impl, "Gemma3MLP", mlp)
    monkeypatch.setattr(
        impl,
        "make_layers",
        lambda count, factory, prefix: (
            0,
            count,
            nn.ModuleList([factory(prefix=f"{prefix}.{i}") for i in range(count)]),
        ),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config.get_text_config()),
        quant_config=None,
        cache_config=None,
    )
    lm = impl.MammothGemma3ForCausalLM(vllm_config=vllm_config, prefix="language_model")
    assert [item.self_attn.layer_type for item in lm.layers] == ["sliding_attention", "full_attention"]
    assert all(isinstance(item.post_feedforward_layernorm, impl.GemmaRMSNorm) for item in lm.layers)
    assert isinstance(lm.norm, impl.GemmaRMSNorm)
    assert lm.lm_head.weight is lm.embed_tokens.weight
    assert lm.gen_head.weight is not lm.gen_embed_tokens.weight
    assert lm.logits_processor.soft_cap == 10.0
    assert lm.gen_logits_processor.soft_cap == 10.0
