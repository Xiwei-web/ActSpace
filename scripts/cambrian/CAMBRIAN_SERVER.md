# Cambrian-S-7B server

Wraps the official code as a local server with the same interface as vLLM (OpenAI Chat Completions), for use by the ActSpace evaluation. Model weights are read-only.

- Server: `scripts/cambrian/cambrian_server.py`
- Start (inside a GPU job, 1 GPU): `scripts/cambrian/serve_cambrian.sh PORT`; listens on 127.0.0.1 only

## Environment

| Item | Version |
|---|---|
| conda environment | `cambrian-s`, python 3.10 |
| torch / torchvision | 2.3.0+cu121 / 0.18.0+cu121 (CUDA runtime 12.1) |
| transformers / tokenizers / accelerate | 4.37.0 / 0.15.0 / 0.23.0 |
| GPU and driver | A100-SXM4-40GB, 570.195.03 |

All package versions are in `env/cambrian-s.pip-freeze.txt`.

**Differences from the official `pyproject.toml`:**
- The official project does not pin torch; 2.3.0 was chosen because the pinned `torchtext==0.18.0` requires it.
- `lm-format-enforcer` 0.11.3 and `interegular` 0.3.3 were added for JSON mode; all other package versions are unchanged.
- The `[gpu]` extras (deepspeed, bitsandbytes, gradio) are not installed.

**Vision tower:** `google/siglip2-so400m-patch14-384` (snapshot e8e4872) in `HF_HOME`. The official loader does not use the vision-tower weights stored in the checkpoint but loads them from HF (`cambrian_arch.py:193`, `llava_next_siglip_encoder.py:578-580`); the two sets of weights differ at the level of bf16 rounding (<= 8e-3).

## Inference path (files and line numbers refer to the official code, commit 058b1b4)

- **Loading:** follows `cambrian/model/builder.py:27-83`, but in bf16 (the official code hard-codes fp16); the weight files are float32.
- **Configuration:** follows `lmms-eval/lmms_eval/models/simple/cambrians.py:141-148`: `si_token_len` 729, `miv_token_len` 64, `image_aspect_ratio` anyres, at most 9 sub-images.
- **Image dispatch:** follows `cambrians.py:261-271`.
  - Exactly 1 image in the whole conversation: `process_images` uses anyres (from `mm_utils.py:289`).
  - More than 1 image: `use_pad=True` (`mm_utils.py:272-287`); each image is padded to a square and resized to 384.
  - When the official `process_images` sees several images it permanently changes `config.image_aspect_ratio` to pad (`mm_utils.py:267-268`). The server resets it to anyres on every request so that the next request is unaffected.
- **Merging and generation:** `prepare_inputs_labels_for_multimodal_for_generation` (`cambrian_arch.py:482-672`), then HF `generate` with `inputs_embeds` (`cambrian_qwen2.py:495-527`).
- **Image placeholders:** each image is written in place as `<image>` plus a newline (`cambrians.py:296-312`); adjacent text segments are joined with newlines.

The **conversation template** is `conv_templates["qwen_2"]` (`conversation.py:575-583`), not the `chat_template` in `tokenizer_config.json`:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
Q<|im_end|>
<|im_start|>assistant
```

A system message in the request replaces the default system text. Stop tokens are 151645 and 151643 (`generation_config.json`).

## Image tokens and context

| Case | Tokens per image | Source |
|---|---|---|
| Multiple images, 640x480 | 588 | measured |
| Multiple images, 480x640 | 594 | measured |
| Exactly 1 image in the whole conversation, 640x480 (anyres) | 2,788 | measured |

- Context limit 32,768. A request with 41 images totals 24,420 tokens and fits.
- With no text between images, at most 55 images fit (32,418 tokens); 56 images give 33,007 tokens and the server returns 400.
- The maximum training length of the model is 11,264 (`tokenizer_model_max_length`); Passive-40 is far longer, and the model's quality at that length is unknown.
- **Two encodings for the same model:** Active turn 0 has only the START image and uses anyres; from turn 1 on there are several images and pad is used. This is the behaviour of the official code and has not been changed.

## Differences from the official behaviour

**Logits only for the last position:** the official forward computes `lm_head` for all prompt positions and casts to fp32 (`cambrian_qwen2.py:449-451`). With 41 images it ran out of GPU memory after requesting 13.80 GiB (HTTP 500). The server defaults to `--prefill-logits last`, computing only the last position during prefill; the logits of the next token are identical, generation is unchanged, and logprobs are saved as usual. `--prefill-logits all` or the request field `x_prefill_logits: "all"` restores the official behaviour.

## Interface

- `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`. Non-streaming only; `stream: true` returns 400.
- Serial generation (one request at a time).
- Parameters: `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`, `seed`, `logprobs`, `top_logprobs`, `response_format`, `stop`. Default sampling values come from `generation_config.json` (0.7 / 0.8 / 20 / 1.05).
- Extra fields: with `return_prompt_text: true` in the request, `x_prompt_text` is returned; replies also carry `x_timing` (`ttft_s`, `total_s`, `queue_s`), `x_image_tokens`, `x_image_mode`, `x_json_mode`, `x_gpu_peak_mem_gib`.
- **JSON mode:** `lm-format-enforcer`'s `JsonSchemaParser({"type": "object"})` is used as a logits processor to enforce the constraint; replies are marked `x_json_mode: "enforced"`.
- **logprobs:** log-softmax of the model's raw logits (before repetition penalty, JSON mask, temperature, top-k and top-p), matching vLLM V1's default definition; `top_logprobs` use the same distribution.
- **Repetition penalty:** generation starts from `inputs_embeds`, so the penalty applies only to generated tokens, not to the prompt. This is the behaviour of the official path and differs from vLLM (which also penalizes prompt tokens).
- **Context:** when prompt plus `max_tokens` exceeds the limit, `max_tokens` is narrowed to the remaining space; when the prompt alone reaches 32,768, the server returns 400 with the same wording as vLLM, and the evaluation records N/A.

## Test results (synthetic images, not evaluation images)

| Test | Input tokens | Peak GPU memory | Result |
|---|---|---|---|
| Single image | 2,825 | 16.09 GiB | answered "Red", total time 2.05 s |
| 9 images, multi-turn | 5,520 | 16.60 GiB | answered "Nine.", first token 0.64 s |
| 41 images | 24,420 | 20.66 GiB | answered "Blue", first token 3.12 s |
| 41 images, official all-position logits | n/a | n/a | out of GPU memory, HTTP 500 |
| JSON mode + logprobs top-5 | 2,825 | 24.65 GiB | reply parsable; 18 tokens in 4.09 s |
| 55 / 56 / 57 images | 32,418 / 33,007 / 33,596 | 31.0 GiB (55 images) | 55 succeeded, 56 and 57 returned 400 |
| Same seed twice | 25 | 24.03 GiB | identical outputs |

**Caveats:**
- Generation is noticeably slower with the JSON constraint on (18 tokens in 4.09 s in the table above, vs. about 0.45 s for 19 tokens without it).
- The JSON test above generated 18 tokens but has only 17 logprobs entries; the missing one is most likely the end token, not verified further.
- Peak GPU memory is measured for a single request. On some nodes torch occasionally could not see the GPU; switching nodes resolved it.
