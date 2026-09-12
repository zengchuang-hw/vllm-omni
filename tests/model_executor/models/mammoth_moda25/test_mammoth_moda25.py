# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig
from vllm.transformers_utils.config import uses_xdrope_dim

from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.mammoth_moda25.mammoth_moda25 import (
    MammothModa25DecoderLayer,
    MammothModa25ForCausalLM,
    MammothModa25ForConditionalGeneration,
)
from vllm_omni.model_executor.models.mammoth_moda25.pipeline import MAMMOTH_MODA25_PIPELINE
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit
from vllm_omni.model_extras.mammoth_moda25 import build_text_to_image_prompt
from vllm_omni.transformers_utils.configs.mammoth_moda25 import MammothModa25Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def config():
    # HunyuanOCR config.json dimensions, with a small generation vocabulary.
    return MammothModa25Config(
        llm_config={
            "model_type": "hunyuan_vl",
            "gen_vocab_size": 16,
            "text_config": {
                "vocab_size": 120818,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "use_qk_norm": True,
                "tie_word_embeddings": True,
                "rope_parameters": {"rope_type": "xdrope", "rope_theta": 10000.0, "xdrope_section": [16, 16, 16, 16]},
            },
            "vision_config": {"out_hidden_size": 1024, "text_hidden_size": 1024},
        },
        gen_token_config={
            "image_start_token_id": 120818,
            "image_token_id": 120819,
            "eol_token_id": 120820,
            "visual_token_start_id": 120821,
            "visual_token_end_id": 120833,
        },
    )


def test_config_roundtrip_and_xdrope(config, tmp_path):
    config.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(restored, MammothModa25Config)
    for cfg in (config, restored, deepcopy(restored)):
        text = cfg.get_text_config()
        assert text.vocab_size == 120834
        assert text.base_vocab_size == 120818
        assert text.hidden_size == 1024
        assert text.head_dim == 128
        assert text.use_qk_norm
        assert text.tie_word_embeddings
        assert uses_xdrope_dim(cfg) == 4
        assert cfg.vision_start_token_id == 120118
        assert cfg.image_newline_token_id == 120121
        assert cfg.tokenizer_class == "PreTrainedTokenizerFast"


def test_reject_qwen_config():
    with pytest.raises(ValueError, match="Hunyuan-OCR"):
        MammothModa25Config(llm_config={"model_type": "mammothmoda2_qwen2_5_vl"})


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages[-1]["role"] == "user"
        assert kwargs["add_generation_prompt"]
        return [3, 4]

    def encode(self, text, **kwargs):
        assert text == "2*1"
        return [5]


def test_prompt_uses_checkpoint_tokens(config):
    result = build_text_to_image_prompt(Tokenizer(), config, "test", height=16, width=32)
    assert result["prompt_token_ids"] == [3, 4, 120818, 5, 120819]
    assert result["additional_information"]["eol_token_id"] == [120820]
    assert result["additional_information"]["ar_width"] == [2]


@pytest.mark.parametrize("height,width", [(0, 32), (16, -1), (17, 32)])
def test_reject_invalid_grid(config, height, width):
    with pytest.raises(ValueError, match="multiples of 16"):
        build_text_to_image_prompt(Tokenizer(), config, "test", height, width)


def test_reject_old_qwen_token_ids(config):
    config.gen_token_config["eol_token_id"] = 152064
    with pytest.raises(ValueError, match="extended generation vocabulary"):
        build_text_to_image_prompt(Tokenizer(), config, "test", 16, 32)


def test_mixed_request_logits_and_row_boundary(config):
    model = MammothModa25ForConditionalGeneration.__new__(MammothModa25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(base_vocab_size=120818)
    info = build_text_to_image_prompt(Tokenizer(), config, "test", 16, 32)["additional_information"]
    model._last_runtime_additional_information = [
        dict(info, generated_len=0),
        dict(info, generated_len=2),
        {"omni_task": ["chat"]},
    ]
    logits = model._apply_t2i_token_constraints(torch.zeros(3, 120834))
    assert torch.where(torch.isfinite(logits[0]))[0].tolist() == list(range(120821, 120834))
    assert torch.where(torch.isfinite(logits[1]))[0].tolist() == [120820]
    assert torch.isfinite(logits[2, :120818]).all()
    assert torch.isneginf(logits[2, 120818:]).all()
    assert model.requires_raw_input_tokens


class Norm(nn.Module):
    def forward(self, hidden, residual=None):
        return hidden if residual is None else (hidden, residual)


class Attention(nn.Module):
    def forward(self, positions, hidden_states, kv_states):
        assert positions.shape[0] == 4
        return hidden_states, None


def test_hunyuan_decoder_routes_generation_expert():
    layer = MammothModa25DecoderLayer.__new__(MammothModa25DecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = Norm()
    layer.post_attention_layernorm = Norm()
    layer.self_attn = Attention()
    layer.mlp = nn.Identity()
    layer.gen_mlp = nn.Linear(2, 2, bias=False)
    layer.gen_mlp.weight.data.copy_(2 * torch.eye(2))
    output, _ = layer(torch.zeros(4, 3), torch.ones(3, 2), None, torch.tensor([False, True, False]))
    torch.testing.assert_close(output, torch.tensor([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]))


def test_ar_to_dit_preserves_hunyuan_hidden_states(config):
    hidden = torch.ones(4, 1024, dtype=torch.bfloat16)
    output = SimpleNamespace(
        prompt_token_ids=[3, 4],
        outputs=[
            SimpleNamespace(
                cumulative_token_ids=[120821, 120820, 120822],
                multimodal_output={"latent": hidden},
            )
        ],
    )
    prompt = build_text_to_image_prompt(Tokenizer(), config, "test", 16, 32)
    converted = ar2dit([output], prompt)[0]["additional_information"]
    assert converted["full_hidden_states"].shape == (4, 1024)
    assert converted["full_hidden_states"].dtype == torch.float32
    assert converted["full_token_ids"] == [3, 4, 120821, 120820]
    assert converted["answer_start_index"] == [2]
    assert converted["image_width"] == [32]


def test_pipeline_stage_contract():
    ar, dit = MAMMOTH_MODA25_PIPELINE.stages
    assert ar.execution_type == StageExecutionType.LLM_AR
    assert ar.model_arch == "MammothModa25ForConditionalGeneration"
    assert ar.engine_output_type == "latent"
    assert dit.model_arch == "MammothModa2DiTPipeline"
    assert dit.input_sources == (0,)
    assert dit.custom_process_input_func.endswith("mammoth_moda2.ar2dit")


def test_weight_loading_and_missing_generation_weights():
    model = MammothModa25ForConditionalGeneration.__new__(MammothModa25ForConditionalGeneration)
    nn.Module.__init__(model)
    lm = MammothModa25ForCausalLM.__new__(MammothModa25ForCausalLM)
    nn.Module.__init__(lm)
    lm.quant_config = None
    lm.embed_tokens = nn.Embedding(4, 2)
    lm.gen_embed_tokens = nn.Embedding(3, 2)
    lm.gen_head = nn.Linear(2, 3, bias=False)
    model.language_model = lm
    base = [("llm_model.model.embed_tokens.weight", torch.ones(4, 2))]
    with pytest.raises(ValueError, match="missing trained generation weights"):
        model.load_weights(iter(base))
    loaded = model.load_weights(
        iter(
            base
            + [
                ("llm_model.model.gen_embed_tokens.weight", torch.full((3, 2), 2.0)),
                ("llm_model.gen_head.weight", torch.full((3, 2), 3.0)),
                ("gen_transformer.unused.weight", torch.empty(1)),
            ]
        )
    )
    assert "language_model.gen_head.weight" in loaded
    torch.testing.assert_close(lm.gen_embed_tokens.weight, torch.full((3, 2), 2.0))
    torch.testing.assert_close(lm.gen_head.weight, torch.full((3, 2), 3.0))


def test_dit_excludes_ocr_image_delimiters(config):
    from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline

    dit = MammothModa2DiTPipeline.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(dit)
    dit.config = config
    ids = [7, 120118, 120120, 120121, 120119, 120821, 120820]
    hidden = torch.arange(7 * 1024).reshape(7, 1024).float()
    text, image = dit._split_ar_conditions(full_hidden_states=hidden, full_token_ids=ids, answer_start_index=5)
    torch.testing.assert_close(text, hidden[:1])
    torch.testing.assert_close(image, hidden[5:])
