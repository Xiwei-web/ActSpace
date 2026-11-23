"""Measure tokens per image with vLLM's own multimodal processor (CPU, no weights loaded).

    PYTHONPATH=src:vllm_plugins <vllm env>/bin/python scripts/vllm_image_tokens.py <model key> [<model key> …] [--images N]

Model parameters come from configs/eval.yaml (hf_overrides, trust_remote_code, dtype, max_model_len as in open_serve.vllm_args).
Output per model: the image placeholder tokens (vLLM's mm_placeholders) for one 640×480 landscape and one 480×640 portrait image,
and the total input length for N images (default 41).
"""
import json
import sys

from PIL import Image

sys.path.insert(0, "src")
from actspace_eval.data import load_eval_config  # noqa: E402


def main():
    args = sys.argv[1:]
    n_many = 41
    if "--images" in args:
        i = args.index("--images")
        n_many = int(args[i + 1])
        del args[i:i + 2]
    import os
    ec = load_eval_config()
    for key in args:
        for k, x in ((ec["models"][key].get("vllm") or {}).get("env") or {}).items():
            os.environ[k] = str(x)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.model_executor.models.registry import ModelRegistry
    out = {}
    for key in args:
        m = ec["models"][key]
        v = m["vllm"]
        mc = ModelConfig(model=m["path"], tokenizer=m["path"], trust_remote_code=bool(v.get("trust_remote_code")), dtype=v.get("dtype", "bfloat16"),
                         max_model_len=min(int(v["max_model_len"]), int(ec["open_models"]["max_model_len_cap"])), seed=0,
                         hf_overrides=v.get("hf_overrides") or {}, limit_mm_per_prompt={"image": int(ec["open_models"]["limit_images_per_prompt"]), "video": 0})
        cls, _ = ModelRegistry.resolve_model_cls(mc.architectures, model_config=mc)
        ph = cls.get_placeholder_str("image", 1)
        proc = MULTIMODAL_REGISTRY.create_processor(mc)
        res = {"architecture": mc.architectures, "placeholder": ph}

        def run(imgs):
            prompt = "".join(f"Photo {k}: {ph}\n" for k in range(len(imgs))) + "Describe."
            items = proc.info.parse_mm_data({"image": imgs}) if hasattr(proc.info, "parse_mm_data") else proc.data_parser.parse_mm_data({"image": imgs})
            r = proc.apply(ProcessorInputs(prompt=prompt, mm_data_items=items), TimingContext())
            pls = r["mm_placeholders"]["image"]
            return [p.get_num_embeds() if hasattr(p, "get_num_embeds") else p.length for p in pls], [p.length for p in pls], len(r["prompt_token_ids"])
        for w, h in ((640, 480), (480, 640)):
            emb, length, total = run([Image.new("RGB", (w, h), (120, 130, 140))])
            res[f"{w}x{h}"] = {"embeds": emb[0], "placeholder_tokens": length[0], "prompt_tokens": total}
        emb, length, total = run([Image.new("RGB", (480, 640) if k % 2 else (640, 480), (120, 130, 140)) for k in range(n_many)])
        res[f"{n_many}_images"] = {"prompt_tokens": total, "placeholder_tokens_sum": sum(length), "max_model_len": mc.max_model_len}
        out[key] = res
        print(json.dumps({key: res}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
