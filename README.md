# Drama 3 review evidence

[PR #10669](https://github.com/vllm-project/vllm-omni/pull/10669) · [final source e5b36206b](https://github.com/vllm-project/vllm-omni/commit/e5b36206b7ea2f96b9b59f048d4701a56d1c77ec) · [merge base 51c99e80b](https://github.com/vllm-project/vllm-omni/commit/51c99e80b7ad84a4cb48e63556e013a94dc85f8a) · [review request](https://github.com/vllm-project/vllm-omni/pull/10669#issuecomment-7008301380)

## Why the prompt boundary changed

On the preceding revision `2b51c3bee`, all six new English and Mandarin two-clause scripts had complete full-clip transcripts. On one Mandarin script, both speaker encoders matched the first clause more closely to a different take's second clause. The text was `[轻声] 今天下午我们去公园散步。[叹气] 记得带上雨伞，天气可能会变。`

We repeated the Mandarin script with seeds 17 through 21. Both arms used the same stage-0 model, defaults, codec and seed. The single-turn input explicitly prefixed the existing speaker-0 tag, which reproduces the prompt rows without the second speaker tag. The arm order alternated by seed. Each row reports cross-clause speaker cosine computed from decoded PCM with the repository's WavLM and CAM++ helpers.

| Seed | WavLM, inserted turn | WavLM, single turn | CAM++, inserted turn | CAM++, single turn |
| --- | ---: | ---: | ---: | ---: |
| 17 | 0.7152 | 0.9415 | 0.1363 | 0.7236 |
| 18 | 0.8824 | 0.9559 | 0.5004 | 0.8442 |
| 19 | 0.9488 | 0.9828 | 0.8251 | 0.8326 |
| 20 | 0.9438 | 0.9371 | 0.5877 | 0.7724 |
| 21 | 0.8177 | 0.9829 | 0.3852 | 0.7955 |

The final code keeps a cue immediately following the sentence punctuation inside that turn. A whitespace separator before the next cue still introduces the new speaker-0 turn. [Matched transcripts](adjacent-cue-comparison/audio-results.json) · [Speaker scores](adjacent-cue-comparison/speakers.json) · [Seed 17 with inserted turn](adjacent-cue-comparison/zh17-multi.wav) · [Seed 17 with single turn](adjacent-cue-comparison/zh17-single.wav)

## Final-head waveform results

All 80 original keys used the same four texts, seeds 0–19, at most eight requests in flight, no reference, and a 1,000-frame limit. All 80 requests returned `stop`. There were zero intervals of at least 10 seconds below threshold. The longest quiet interval anywhere was 2.15 s and the longest tail was 0.35 s. Whisper large-v3 made 9 word errors in 1,440 reference words. The requested final clauses were present in each of the four previously failing takes. [Timing for all 80](cohort80.json) · [All 80 transcriptions](asr-final-80.json)

| Original failing take | Final tail (s) |
| --- | ---: |
| [Prompt 0, seed 6](four-original-cases/p0-s06.wav) | 0.05 |
| [Prompt 0, seed 16](four-original-cases/p0-s16.wav) | 0.30 |
| [Prompt 0, seed 18](four-original-cases/p0-s18.wav) | 0.20 |
| [Prompt 3, seed 7](four-original-cases/p3-s07.wav) | 0.00 |

The six new event-led scripts all returned `stop`. WavLM and CAM++ both matched each first clause most closely to its own second clause among the three same-language takes. The score is cosine similarity. Whisper's full-clip text matches the script for five takes after ignoring capitalization and punctuation. On zh3, full-clip Whisper heard 前面 at the end and isolated-clause Whisper heard the requested 见面. On zh2, the full-clip and the isolated-clause transcripts also differ on one phrase. An independent Qwen3-ASR-1.7B transcript contains all requested spoken words across the six takes, including the intended last phrases of zh2 and zh3. Qwen also transcribes vocalizations on zh1 and zh3 as 啊 and 哼. [Independent ASR](independent-asr.json) · [Full and split Whisper transcripts](audio-results.json) · [Speaker comparison matrices](speakers.json)

| Clip | Duration (s) | Final quiet (s) | WavLM | CAM++ |
| --- | ---: | ---: | ---: | ---: |
| [en1](en1.wav) | 6.50 | 0.18 | 0.973 | 0.733 |
| [en2](en2.wav) | 6.32 | 0.14 | 0.796 | 0.732 |
| [en3](en3.wav) | 5.02 | 0.00 | 0.893 | 0.746 |
| [zh1](zh1.wav) | 6.97 | 0.16 | 0.942 | 0.724 |
| [zh2](zh2.wav) | 5.94 | 0.28 | 0.942 | 0.828 |
| [zh3](zh3.wav) | 5.71 | 0.14 | 0.902 | 0.699 |

### Matched sampler timing

One CUDA-event interval over 100 warmed graph replays on the same L20X for each checkout. Each replay samples a batch of eight sets of fixed BF16 logits with one main draw and nine codebook draws. The base (width 64, main top-k 30 / top-p 0.9) took 303.89 µs per batch. Final head (width 4096, full main draw) took 299.21 µs. The measurements cover the sampler only. Run `python "$EVIDENCE/review-bench.py" base` from the base checkout and `python "$EVIDENCE/review-bench.py" head` from the final checkout. [Timing output](logs/matched-bench.log)

## Unit-test evidence at final head

| Selection | Passed | Failed | Skipped | Deselected | Collection errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fish model and serving reference cases | 217 | 0 | 0 | 0 | 0 |
| H100 markers on L20X | 118 | 0 | 7 | 17,820 | 0 |
| Two Buildkite contract files, each checkout | 434 | 0 | 6 | 0 | 0 |
| Full CPU selection without a visible GPU, merge base | 0 | 0 | 30 | 6,860 | 2 |
| Full CPU selection without a visible GPU, final head | 0 | 0 | 30 | 6,863 | 2 |

The exact R2-01 guard collected 1,439 of 13,198 nodes, with 11,759 deselected. The no-visible-GPU CPU runs both stopped during collection in `test_hunyuan_image3_adapter_cuda.py` and `test_shared_postload_cuda.py`. Both modules invoke a skip outside a test. The full visible-GPU CPU runs are recorded separately. [Collection](logs/guard-head.log) · [Fish](logs/fish-final.log) · [H100 selector](logs/cuda-H100.log) · [Buildkite base](logs/isolated-checks-base.log) · [Buildkite head](logs/isolated-checks-head.log) · [CPU collection, base](logs/cpu-base.log) · [CPU collection, head](logs/cpu-head.log)

## Reproduction

Linux x86-64, NVIDIA L20X (SM 8.9), driver 570.133.20, Python 3.12.13, pytest 9.1.1, vLLM 0.34.0, PyTorch 2.14.0+cu130, torchaudio 2.11.0+cu130, Triton 3.8.0, Transformers 5.19.0. Base and head used the same Python environment, with each source checkout first on `PYTHONPATH` and Hugging Face offline. The locally installed vLLM-Omni package metadata predates these checkouts. Source for the experiments was exported from the revisions named above.

The six-script harness uses the final source's V2 stage 0, marks it as the pipeline's output and exports its acoustic codes. Its decoder loads all DAC parameters with strict key matching, then writes 44.1 kHz PCM. Run the scripts from the source checkout in the order below, using one visible GPU. Point each variable at the corresponding local snapshot.

| Variable | Snapshot |
| --- | --- |
| `DRAMA3_MODEL` | `fishaudio/drama-3@b2b5ee998034b2975720906c815e7fb0e9f007bd`, codec.pth SHA-256 `74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f` |
| `WHISPER_MODEL` | `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1` |
| `WAVLM_MODEL` | `microsoft/wavlm-base-plus-sv@feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` |
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5` |
| `CAMP_MODEL` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512@29e01c4e8d000f4bcd70751be16fa94bf3d85a18`, campplus.onnx SHA-256 `a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73` |

```sh
export PYTHONPATH="$PWD:$PYTHONPATH"
python "$EVIDENCE/review-stage0.py"
python "$EVIDENCE/review-audio.py"
python "$EVIDENCE/review-speakers.py"
python "$EVIDENCE/second-asr.py"
python "$EVIDENCE/run-80.py"
python "$EVIDENCE/score-80.py"
```

For the six new takes, the split uses a quiet 100 ms window near the text-length estimate. Independent ASR on each side checks the clause boundary. Cosines compare each first clause against all three second clauses in its language. These unconditioned takes have no reference speaker labels. Tail silence uses 20 ms windows. The original 80-case evaluation uses its historical 50 ms windows and counts a failure at 10 seconds of quiet audio or generation length exhaustion. Both methods define quiet relative to the 95th-percentile RMS of the take.

```sh
pytest -sv tests/ -m 'core_model and cpu'
pytest -sv tests/engine/ tests/model_executor/ tests/worker_v2/ tests/worker/ -m 'core_model and cuda and H100' --run-level core_model
pytest -sv tests/engine/ tests/model_executor/ tests/worker_v2/ tests/worker/ -m 'core_model and cuda and L4' --run-level core_model
python -m pytest -q tests/entrypoints/openai_api/test_fish_omni_reference_codes.py tests/model_executor/models/fish_omni
python -m pytest -q tests/buildkite/test_amd_ready_r2_01.py tests/buildkite/test_amd_pipeline.py
bash .buildkite/amd/scripts/assert-pytest-collection.sh 1439 tests/model_executor/ -m 'core_model and rocm and MI325 and not nvidia_only and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)' --run-level core_model
```

The two Buildkite test files use their own empty `/tmp` per checkout, including for subprocesses that use hard-coded output paths. The native AMD job runs the selected model-executor tests with `--num-shards=3` and one of `--shard-id=0`, `1` and `2` per worker. The native H100, L4, ROCm and NPU runs are controlled by project CI. A CUDA collection check establishes the selected node count. [ROCm job](https://github.com/vllm-project/vllm-omni/blob/e5b36206b7ea2f96b9b59f048d4701a56d1c77ec/.buildkite/amd/test-amd-ready.yml#L652) · [CUDA jobs](https://github.com/vllm-project/vllm-omni/blob/e5b36206b7ea2f96b9b59f048d4701a56d1c77ec/.buildkite/cuda/test-ready.yml#L133) · [NPU jobs](https://github.com/vllm-project/vllm-omni/blob/e5b36206b7ea2f96b9b59f048d4701a56d1c77ec/.buildkite/npu/test-npu-ready.yml)

