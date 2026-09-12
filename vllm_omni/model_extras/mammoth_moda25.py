# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the AR image grid from the trained checkpoint's token IDs."""


def build_text_to_image_prompt(tokenizer, config, prompt, height=1024, width=1024):
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("Image height and width must be positive multiples of 16")
    token_config = getattr(config, "gen_token_config", None)
    required = (
        "image_start_token_id",
        "image_token_id",
        "eol_token_id",
        "visual_token_start_id",
        "visual_token_end_id",
    )
    if not isinstance(token_config, dict) or any(key not in token_config for key in required):
        raise ValueError(f"Checkpoint config requires gen_token_config with {required}")
    start = config.llm_config.gen_vocab_start_index
    end = start + config.llm_config.gen_vocab_size
    ids = {key: int(token_config[key]) for key in required}
    if any(not start <= value < end for value in ids.values()):
        raise ValueError("All gen_token_config IDs must belong to the extended generation vocabulary")
    if ids["visual_token_start_id"] > ids["visual_token_end_id"]:
        raise ValueError("Invalid visual token range")
    if any(ids["visual_token_start_id"] <= ids[key] <= ids["visual_token_end_id"] for key in required[:3]):
        raise ValueError("Image delimiters and EOL must be outside the visual code range")
    ar_width, ar_height = width // 16, height // 16
    token_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful image generator."}, {"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = list(token_ids) + [ids["image_start_token_id"]]
    token_ids += tokenizer.encode(f"{ar_width}*{ar_height}", add_special_tokens=False)
    token_ids += [ids["image_token_id"]]
    return {
        "prompt_token_ids": token_ids,
        "additional_information": {
            "omni_task": ["t2i"],
            "ar_width": [ar_width],
            "ar_height": [ar_height],
            "image_height": [height],
            "image_width": [width],
            **{key: [ids[key]] for key in required[2:]},
        },
    }
