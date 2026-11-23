"""ActSpace vLLM plugins.

vLLM calls register() in every process via the vllm.general_plugins entry point. The plugins are only loaded when the server
process has vllm_plugins/ on its PYTHONPATH (open_serve.server_env: model config vllm.plugins: true). None of vLLM's built-in
models is changed.

- LlavaOnevisionBaseOnly: LLaVA-OneVision with anyres high-resolution tiling disabled; every image uses the base resolution only
  (see llava_ov_base.py). Usage: hf_overrides {"architectures": ["LlavaOnevisionBaseOnly"], "image_grid_pinpoints": [[384, 384]]}.
- MiniCPM-V 4.5 tokenizer attributes (when the environment variable ACTSPACE_MINICPMV_TOKENIZER=1): the official
  tokenization_minicpmv_fast.MiniCPMVTokenizerFast only adds the names and id attributes of a few special tokens (im_start_id etc.)
  to the Qwen2 tokenizer. transformers 5.5.4 no longer loads this class via auto_map, and the official processor
  (processing_minicpmv.py) fails when it reads tokenizer.im_start_id. Here the same attributes are added verbatim to the tokenizer
  base class (only those that are missing); vocabulary, encoding and template are unchanged.
"""
import os

MINICPMV_TOKENS = {"im_start": "<image>", "im_end": "</image>", "ref_start": "<ref>", "ref_end": "</ref>", "box_start": "<box>",
                   "box_end": "</box>", "quad_start": "<quad>", "quad_end": "</quad>", "slice_start": "<slice>", "slice_end": "</slice>",
                   "im_id_start": "<image_id>", "im_id_end": "</image_id>"}


def patch_minicpmv_tokenizer():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as B

    for name, tok in MINICPMV_TOKENS.items():
        if not hasattr(B, name):
            setattr(B, name, tok)
    props = {"eos_id": lambda s: s.eos_token_id, "bos_id": lambda s: s.bos_token_id, "unk_id": lambda s: s.unk_token_id,
             "newline_id": lambda s: s.convert_tokens_to_ids("\n")}
    for name in ("im_start", "im_end", "slice_start", "slice_end", "im_id_start", "im_id_end"):
        props[f"{name}_id"] = (lambda n: lambda s: s.convert_tokens_to_ids(getattr(s, n)))(name)
    for name, f in props.items():
        if not hasattr(B, name):
            setattr(B, name, property(f))
    for name in ("escape", "unescape"):
        if not hasattr(B, name):
            setattr(B, name, staticmethod(lambda text: text))


def register():
    from vllm import ModelRegistry

    if "LlavaOnevisionBaseOnly" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("LlavaOnevisionBaseOnly", "actspace_vllm_plugins.llava_ov_base:LlavaOnevisionBaseOnly")
    if os.environ.get("ACTSPACE_MINICPMV_TOKENIZER") == "1":
        patch_minicpmv_tokenizer()
