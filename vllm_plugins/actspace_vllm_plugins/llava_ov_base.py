"""LLaVA-OneVision with the base resolution only for every image (following the precedent of tile limit 1 for InternVL).

The original (anyres_max_9) encodes every image as the base view (whole image resized to 384×384, 27×27 = 729 features) plus
high-resolution tiles (about 2,200 more for 640×480). Here only the base view is kept, followed by one image_newline as in the
model's own "base view only" path: 730 tokens per image, the same for landscape and portrait images.
Weights, vision tower, projector and language model are unchanged; the tile features are simply not fed to the language model.

- Processor: text and images are passed to the transformers processor separately (<image> in the text is not expanded); vLLM
  replaces the placeholder token count according to get_num_image_tokens. image_grid_pinpoints is set to [[384, 384]]
  (hf_overrides), so the processor computes only 1 extra tile (discarded afterwards) and skips the vision compute of the others.
- Model: _process_image_input keeps only view 0 (the base view) of every image, followed by image_newline.
"""
from __future__ import annotations

import torch
from transformers import BatchFeature

from vllm.model_executor.models.llava_onevision import (LlavaOnevisionDummyInputsBuilder, LlavaOnevisionForConditionalGeneration,
                                                         LlavaOnevisionMultiModalProcessor, LlavaOnevisionProcessingInfo)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize


class BaseOnlyProcessingInfo(LlavaOnevisionProcessingInfo):
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        vc = self.get_hf_config().vision_config
        return (vc.image_size // vc.patch_size) ** 2 + 1      # base view + image_newline

    def get_image_size_with_most_features(self):
        vc = self.get_hf_config().vision_config
        return ImageSize(width=vc.image_size, height=vc.image_size)


class BaseOnlyMultiModalProcessor(LlavaOnevisionMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
        mm_data = dict(mm_data)
        images = mm_data.pop("images", [])
        if mm_data.get("videos"):
            raise ValueError("LlavaOnevisionBaseOnly only supports images")
        text = super(LlavaOnevisionMultiModalProcessor, self)._call_hf_processor(prompt=prompt, mm_data={}, mm_kwargs=mm_kwargs, tok_kwargs=tok_kwargs)
        if not images:
            return text
        proc = self.info.get_hf_processor()
        out = super(LlavaOnevisionMultiModalProcessor, self)._call_hf_processor(
            prompt=proc.image_token * len(images), mm_data={"images": images}, mm_kwargs=mm_kwargs, tok_kwargs=tok_kwargs)
        return BatchFeature(dict(text, **{k: v for k, v in out.items() if k in ("pixel_values", "image_sizes")}))

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs) -> bool:
        return False


@MULTIMODAL_REGISTRY.register_processor(BaseOnlyMultiModalProcessor, info=BaseOnlyProcessingInfo, dummy_inputs=LlavaOnevisionDummyInputsBuilder)
class LlavaOnevisionBaseOnly(LlavaOnevisionForConditionalGeneration):
    def _process_image_input(self, image_input):
        if image_input["type"] == "image_embeds":
            return image_input["data"]
        patches = self._process_image_pixels(image_input)
        return [torch.cat((p[0], self.image_newline[None].to(p.device, p.dtype)), dim=0) for p in patches]
