# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MammothModa2.5: Hunyuan-OCR AR with the MammothModa2 DiT."""

from copy import deepcopy

from transformers import AutoConfig
from vllm.transformers_utils.configs.hunyuan_vl import HunYuanVLConfig

from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config


class MammothModa25HunyuanConfig(HunYuanVLConfig):
    model_type = "mammothmoda25_hunyuan_vl"

    def __init__(self, *, gen_vocab_size=None, gen_vocab_start_index=None, moe_type=None, **kwargs):
        super().__init__(**deepcopy(kwargs))
        text = self.text_config
        gen_vocab_size = int(gen_vocab_size if gen_vocab_size is not None else getattr(text, "gen_vocab_size", 32800))
        moe_type = moe_type if moe_type is not None else getattr(text, "moe_type", "ffn")
        base = int(getattr(text, "base_vocab_size", text.vocab_size))
        start = base if gen_vocab_start_index is None else int(gen_vocab_start_index)
        if start != base or gen_vocab_size <= 0:
            raise ValueError("Generation vocabulary must be nonempty and immediately follow the base vocabulary")
        if moe_type not in ("ffn", "none"):
            raise ValueError("MammothModa2.5 supports moe_type='ffn' or 'none'")
        text.base_vocab_size = base
        text.extra_gen_vocab = True
        text.gen_vocab_start_index = start
        text.gen_vocab_size = int(gen_vocab_size)
        text.vocab_size = start + int(gen_vocab_size)
        text.moe_type = moe_type
        rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None)
        if rope and "xdrope_section" in rope:
            text.rope_scaling = deepcopy(rope)
            text.xdrope_section = list(rope["xdrope_section"])
        self.image_start_token_id = getattr(self, "image_start_token_id", self.im_start_id)
        self.image_end_token_id = getattr(self, "image_end_token_id", self.im_end_id)
        # These are read on both the VL config and the text config.
        self.gen_vocab_start_index = start
        self.gen_vocab_size = int(gen_vocab_size)
        self.extra_gen_vocab = True
        self.moe_type = moe_type


class MammothModa25Config(Mammothmoda2Config):
    model_type = "mammothmoda25"

    def __init__(self, *, llm_config=None, **kwargs):
        if llm_config is not None:
            llm_config = deepcopy(llm_config)
            if llm_config.get("model_type") not in ("hunyuan_vl", "mammothmoda25_hunyuan_vl"):
                raise ValueError("MammothModa2.5 requires a Hunyuan-OCR llm_config")
            llm_config["model_type"] = MammothModa25HunyuanConfig.model_type
        super().__init__(llm_config=llm_config, **kwargs)
        self.architectures = ["MammothModa25ForConditionalGeneration"]
        # Use the checkpoint's Hunyuan tokenizer.json and processor assets.
        self.tokenizer_class = "PreTrainedTokenizerFast"

    @property
    def video_token_id(self):
        return -1  # Hunyuan-OCR has no video input tokens.

    @property
    def vision_start_token_id(self):
        return self._require_llm_config().image_start_token_id

    @property
    def vision_end_token_id(self):
        return self._require_llm_config().image_end_token_id

    @property
    def image_newline_token_id(self):
        return self._require_llm_config().im_newline_id

    @property
    def xdrope_section(self):
        return getattr(self._require_llm_config().text_config, "xdrope_section", None)


AutoConfig.register(MammothModa25HunyuanConfig.model_type, MammothModa25HunyuanConfig)
AutoConfig.register(MammothModa25Config.model_type, MammothModa25Config)
