# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mammoth DiT composition with a generation-aware Gemma 3 backbone."""

from copy import deepcopy

from transformers import AutoConfig, Gemma3Config

from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config


class MammothModa25GemmaARConfig(Gemma3Config):
    model_type = "mammothmoda25_gemma3"

    def __init__(self, *, gen_vocab_size=None, gen_vocab_start_index=None, moe_type=None, **kwargs):
        super().__init__(**deepcopy(kwargs))
        text = self.text_config
        size = int(gen_vocab_size if gen_vocab_size is not None else getattr(text, "gen_vocab_size", 32800))
        base = int(getattr(text, "base_vocab_size", text.vocab_size))
        start = base if gen_vocab_start_index is None else int(gen_vocab_start_index)
        mode = moe_type if moe_type is not None else getattr(text, "moe_type", "ffn")
        if start != base or size <= 0:
            raise ValueError("Generation vocabulary must be nonempty and immediately follow the base vocabulary")
        if mode not in ("ffn", "none"):
            raise ValueError("Gemma AR supports moe_type='ffn' or 'none'")
        if getattr(text, "use_bidirectional_attention", False):
            raise ValueError("The AR backbone requires causal text attention")
        text.base_vocab_size = base
        text.vocab_size = start + size
        text.extra_gen_vocab = True
        text.gen_vocab_start_index = self.gen_vocab_start_index = start
        text.gen_vocab_size = self.gen_vocab_size = size
        text.moe_type = self.moe_type = mode


class MammothModa25GemmaConfig(Mammothmoda2Config):
    model_type = "mammothmoda25_gemma"

    def __init__(self, *, llm_config=None, **kwargs):
        if llm_config is not None:
            llm_config = deepcopy(llm_config)
            if llm_config.get("model_type") not in ("gemma3", MammothModa25GemmaARConfig.model_type):
                raise ValueError("This pipeline requires a multimodal Gemma 3 llm_config")
            llm_config["model_type"] = MammothModa25GemmaARConfig.model_type
        super().__init__(llm_config=llm_config, **kwargs)
        self.architectures = ["MammothModa25GemmaForConditionalGeneration"]
        self.tokenizer_class = "PreTrainedTokenizerFast"

    @property
    def image_token_id(self):
        return self._require_llm_config().image_token_index

    @property
    def video_token_id(self):
        return -1

    @property
    def vision_start_token_id(self):
        return self._require_llm_config().boi_token_index

    @property
    def vision_end_token_id(self):
        return self._require_llm_config().eoi_token_index


AutoConfig.register(MammothModa25GemmaARConfig.model_type, MammothModa25GemmaARConfig)
AutoConfig.register(MammothModa25GemmaConfig.model_type, MammothModa25GemmaConfig)
