# MammothModa2.5 with Gemma 3 AR

This experimental variant replaces the Hunyuan-OCR AR backbone with multimodal
Gemma 3 and retains the MammothModa2 AR -> DiT/VAE contract. It is not a claim
of compatibility with an official trained Mammoth/Gemma checkpoint.

The implementation follows vLLM's native Gemma 3 modules and the
[Gemma 3 architecture](https://huggingface.co/docs/transformers/model_doc/gemma3).
Use a complete multimodal configuration such as
[google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it).
Dimensions are read from the checkpoint. Gemma 1, Gemma 2, Gemma 3n and
text-only Gemma 3 checkpoints are not supported by this variant.

## Model Structure

- AR vision: Gemma 3's SigLIP vision tower and multimodal projector.
- AR language: native Gemma 3 local/global attention and RoPE, Q/K norms,
  four decoder norms, Gemma final RMSNorm, embedding normalization by
  `sqrt(hidden_size)`, tied base embeddings/head when configured, and optional
  final-logit softcapping. Vision embeddings are not multiplied by the token
  embedding normalizer.
- Generation: separate image-token embeddings/head and an optional generation
  FFN using Gemma's activation. Raw token IDs are retained for expert routing.
- Bridge: the existing `mammoth_moda2.ar2dit` passes hidden states and token
  IDs, omitting the final look-ahead token. The DiT uses the configured Gemma
  hidden dimension and excludes Gemma image placeholder/delimiter tokens from
  text conditioning.

The initial execution path is offline text-to-image in eager mode. DiT uses
the same `LLM_GENERATION` runner as MammothModa2. LoRA, speculative decoding
and video input are not supported. GPU generation has not been validated
without a trained combined checkpoint.

## Trained Checkpoint

Do not concatenate unchanged Gemma and Mammoth checkpoints. Generation
embeddings, generation head, generation experts and DiT conditioning modules
must be trained together for this backbone. Missing generation parameters
produce an explicit loading error.

Configure a combined checkpoint as follows:

1. Set top-level `model_type` to `mammothmoda25_gemma` and `architectures` to
   `["MammothModa25GemmaForConditionalGeneration"]`.
2. Replace `llm_config` with the full Gemma 3 multimodal config, including
   `text_config`, `vision_config`, `image_token_index`, `boi_token_index`,
   `eoi_token_index` and `mm_tokens_per_image`. Its `model_type` can be `gemma3`
   or `mammothmoda25_gemma3`.
3. Add `gen_vocab_size` (default 32800) and `moe_type` (`ffn` or `none`) to
   `llm_config`. `gen_vocab_start_index` must equal the original base vocabulary
   size. Saved configs preserve `base_vocab_size`, so round trips do not grow
   the vocabulary a second time.
4. Retain the trained Mammoth `gen_dit_config`, `gen_vae_config`, optional
   `gen_image_condition_refiner_config`, `gen_condition_mode`, `gen_axes_*`
   and `gen_transport_config` fields. Conditioning weights must match Gemma's
   hidden size.
5. Supply top-level `gen_token_config` with `image_start_token_id`,
   `image_token_id`, `eol_token_id`, `visual_token_start_id`, and
   `visual_token_end_id`. All belong to the extended generation vocabulary;
   the first three must be outside the visual code range. These are trained
   generation tokens, not Gemma's image-input placeholders.
6. Include Gemma's tokenizer, chat template and processor assets, extended
   with the trained generation tokens. Do not use the Hunyuan or Qwen tokenizer.

Expected checkpoint layout (modern Hugging Face Gemma 3 names under
`llm_model`, with added generation parameters):

| Checkpoint prefix | Runtime prefix |
| --- | --- |
| `llm_model.model.language_model.*` | `language_model.*` |
| `llm_model.model.vision_tower.*` | `vision_tower.*` |
| `llm_model.model.multi_modal_projector.*` | `multi_modal_projector.*` |
| `llm_model.lm_head.*` | `language_model.lm_head.*` |
| `llm_model.gen_head.*` | `language_model.gen_head.*` |
| `llm_model.model.language_model.gen_embed_tokens.*` | `language_model.gen_embed_tokens.*` |
| `llm_model.model.language_model.layers.N.gen_mlp.*` | `language_model.layers.N.gen_mlp.*` |
| `gen_transformer.*`, `gen_vae.*`, `gen_image_condition_refiner.*` | Existing DiT/VAE names |

## Run

From the repository root:

```bash
python -m examples.offline_inference.mammoth_moda25_gemma.text_to_image \
  --model /path/to/trained-mammoth-gemma \
  --prompt "A red bicycle beside a green door" \
  --height 1024 --width 1024 --output outputs/mammoth_gemma.png
```

The default deployment config is `vllm_omni/deploy/mammoth_moda25_gemma.yaml`.
Adjust device assignments and memory fractions for the checkpoint size.
Image dimensions must be positive multiples of 16. The example folds the
image-generation instruction into the user turn and uses the checkpoint's
Gemma chat template, followed by its generation-grid tokens. AR sampling
allocates one token per grid cell, one EOL per row and one look-ahead token.
Use this dedicated example rather than the Hunyuan/Qwen prompt builders.
