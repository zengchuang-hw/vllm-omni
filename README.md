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

## Reproduction

Linux x86-64, NVIDIA L20X (SM 8.9), driver 570.133.20, Python 3.12.13, pytest 9.1.1, vLLM 0.34.0, PyTorch 2.14.0+cu130, torchaudio 2.11.0+cu130, Triton 3.8.0, Transformers 5.19.0. Base and head used the same Python environment, with each source checkout first on `PYTHONPATH` and Hugging Face offline. The locally installed vLLM-Omni package metadata predates these checkouts. Source for the experiments was exported from the revisions named above.

The six-script harness uses the final source's V2 stage 0, marks it as the pipeline's output and exports its acoustic codes. Its decoder loads all DAC parameters with strict key matching, then writes 44.1 kHz PCM. Run the scripts from the source checkout in the order below, using one visible GPU. Point each variable at the corresponding local snapshot.

| Variable | Snapshot |
| --- | --- |
| `DRAMA3_MODEL` | `fishaudio/drama-3@b2b5ee998034b2975720906c815e7fb0e9f007bd`, codec.pth SHA-256 `74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f` |
| `WHISPER_MODEL` | `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1` |
| `WAVLM_MODEL` | `microsoft/wavlm-base-plus-sv@feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` |
| `CAMP_MODEL` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512@29e01c4e8d000f4bcd70751be16fa94bf3d85a18`, campplus.onnx SHA-256 `a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73` |

```sh
export PYTHONPATH="$PWD:$PYTHONPATH"
python "$EVIDENCE/review-stage0.py"
python "$EVIDENCE/review-audio.py"
python "$EVIDENCE/review-speakers.py"
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

