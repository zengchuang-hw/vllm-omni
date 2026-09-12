# MammothModa2.5 with Hunyuan-OCR

This is an experimental architecture requested for this repository, not an
implementation verified against an official MammothModa2.5 release checkpoint.
It follows MammothModa2-Preview's AR -> DiT/VAE topology and replaces the
Qwen2.5-VL backbone with Hunyuan-OCR.

## Architecture

- AR: Hunyuan-OCR vision tower, Hunyuan attention with Q/K normalization and
  XDRoPE, plus Mammoth's separate image-token embeddings, generation head and
  generation FFN experts. The generation vocabulary immediately follows the
  base Hunyuan vocabulary.
- Bridge: the existing `mammoth_moda2.ar2dit` transfers prompt/generated token
  IDs and all corresponding hidden states. The final look-ahead token is
  excluded because its hidden state has not been computed.
- DiT/VAE: the existing MammothModa2 implementation. The caption projection
  and optional image condition refiner use the configured AR hidden dimension.
  Hunyuan-OCR's released text hidden dimension is 1024.

The pipeline retains MammothModa2's `LLM_GENERATION` execution type for DiT.
The initial supported path is offline text-to-image with eager execution.
LoRA, speculative decoding and video inputs are not supported by this adapter.

## Checkpoint Contract

Supply a **trained combined checkpoint**. Renaming Hunyuan-OCR or combining its
unchanged weights with MammothModa2-Preview does not produce a trained image
generator. Generation embeddings, generation head, generation FFNs and the
DiT conditioning projections must be trained for this backbone. Missing
generation weights cause an explicit loading error.

Start the top-level configuration from MammothModa2-Preview's configuration:

1. Set `model_type` to `mammothmoda25` and `architectures` to
   `["MammothModa25ForConditionalGeneration"]`.
2. Replace `llm_config` with the complete
   [Hunyuan-OCR configuration](https://huggingface.co/tencent/HunyuanOCR/blob/main/config.json),
   preserving its text/vision dimensions, RoPE parameters and special tokens.
   Either `hunyuan_vl` or `mammothmoda25_hunyuan_vl` is accepted for its
   `model_type`.
3. Add `gen_vocab_size` (default 32800), `gen_vocab_start_index` (120818 for
   the released OCR vocabulary), and `moe_type` (`ffn`, or `none` when trained
   without generation experts) to `llm_config`. Serialization preserves the
   base vocabulary size so reloading does not extend it twice.
4. Retain the trained `gen_dit_config`, `gen_vae_config`,
   `gen_image_condition_refiner_config`, `gen_condition_mode`,
   `gen_axes_dim_rope`, `gen_axes_lens` and `gen_transport_config`.
5. Add top-level `gen_token_config` with the trained tokenizer's integer
   `image_start_token_id`, `image_token_id`, `eol_token_id`,
   `visual_token_start_id`, and `visual_token_end_id`. All must be in the
   generation vocabulary; the first three must be outside the visual code
   range. These describe **generation** tokens, not OCR input placeholders.

Keep Hunyuan's `tokenizer.json`, `tokenizer_config.json`, chat template and
processor assets in the checkpoint directory, with the trained generation
tokens added. Do not use the MammothU/Qwen tokenizer or its fixed token IDs.

Expected weight prefixes:

| Checkpoint prefix | Runtime module |
| --- | --- |
| `llm_model.vit.vit.*` or `llm_model.vit.*` | `visual.*` |
| `llm_model.model.*` | `language_model.*` |
| `llm_model.model.gen_embed_tokens.*` | `language_model.gen_embed_tokens.*` |
| `llm_model.lm_head.*` | `language_model.lm_head.*` |
| `llm_model.gen_head.*` | `language_model.gen_head.*` |
| `llm_model.model.layers.N.gen_mlp.*` | `language_model.layers.N.gen_mlp.*` |
| `gen_transformer.*`, `gen_vae.*`, `gen_image_condition_refiner.*` | Existing DiT/VAE modules |

Hunyuan's base embedding/head tying follows `text_config.tie_word_embeddings`.
Generation embeddings and head are separate, as in MammothModa2.

## Run

From the repository root, with trained weights available:

```bash
python -m examples.offline_inference.mammoth_moda25.text_to_image \
  --model /path/to/trained-mammoth-moda25 \
  --prompt "A red bicycle beside a green door" \
  --height 1024 --width 1024 \
  --output outputs/mammoth_moda25.png
```

The example uses `vllm_omni/deploy/mammoth_moda25.yaml`, builds the prompt using
the checkpoint's chat template and generation IDs, and requests one token per
16x16 image cell, one EOL per row, and one final look-ahead token. Image
dimensions must be positive multiples of 16. Adjust device allocation and
memory fractions in the deploy configuration for your trained checkpoint.

This dedicated example is required: MammothModa2's shared prompt builder uses
Qwen chat formatting and token IDs and must not be used for this architecture.
