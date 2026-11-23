"""Format-adapted chat templates and the image/text interleaving check.

Templates are rendered the transformers way (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, generation tags); vLLM passes
the openai format: content is always a list, text is {"type": "text", "text": …}, images are {"type": "image"}.
"""
import json
from pathlib import Path

import jinja2
import pytest
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

from actspace_eval import content_order
from actspace_eval.report_open_full import gate_result, tokens_by_orientation

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "configs" / "chat_templates"


class Generation(Extension):
    """transformers' {% generation %}…{% endgeneration %} (here the content is just emitted as is)."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def render(name, messages, **kw):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, **k: json.dumps(x, **k)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.exceptions.TemplateError(m))
    return env.from_string((T / name).read_text()).render(messages=messages, **kw)


def as_list(msgs):
    """vLLM openai format: string content becomes a list with one text part."""
    return [{**m, "content": [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]} for m in msgs]


TEXT_CONV = [
    {"role": "system", "content": "You are answering questions about an indoor scene.\n\nReply with JSON."},
    {"role": "user", "content": "Questions\nQ1 (multiple choice) Where is the sofa?\n  A. front\n  B. back\nAnswer all questions."},
    {"role": "assistant", "content": '{"action": "move", "move": "forward"}'},
    {"role": "user", "content": "Move 1: \"forward\". Pose relative to START: forward 0.5 m.\nAnswer or move."},
]
IMG = {"type": "image"}


def image_conv():
    return [{"role": "system", "content": [{"type": "text", "text": "System prompt."}]},
            {"role": "user", "content": [{"type": "text", "text": "Questions\nQ1 ...\n\nPhotographs (3):\nSTART photo_111147\n"}, IMG,
                                         {"type": "text", "text": "\nphoto_2\n"}, IMG, {"type": "text", "text": "\nphoto_3\n"}, IMG,
                                         {"type": "text", "text": "\nAnswer all questions."}]}]


# ---------------- LLaVA-OneVision ----------------

@pytest.mark.parametrize("gen", [True, False])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_llava_text_only_identical(gen, n):
    msgs = as_list(TEXT_CONV[:n])
    off = render("official/llava_onevision_7b.jinja", msgs, add_generation_prompt=gen)
    assert render("llava_onevision_interleaved.jinja", msgs, add_generation_prompt=gen) == off
    assert render("llava_onevision_interleaved.jinja", TEXT_CONV[:n], add_generation_prompt=gen) == off   # same for string content


def test_llava_images_in_place():
    msgs = image_conv()
    new = render("llava_onevision_interleaved.jinja", msgs, add_generation_prompt=True)
    off = render("official/llava_onevision_7b.jinja", msgs, add_generation_prompt=True)
    assert "<|im_start|>user <image><image><image>\n" in off                   # official: all images first
    assert "START photo_111147\n<image>\nphoto_2\n<image>\nphoto_3\n<image>\nAnswer" in new.replace("\n\n", "\n")
    assert new.count("<image>") == 3
    assert new.replace("<image>", "") == off.replace("<image>", "")             # identical without placeholders: only the order changes, not the content
    # the text before each placeholder matches the request
    pieces = new.split("<image>")
    assert pieces[0].endswith("START photo_111147\n") and pieces[1].endswith("photo_2\n") and pieces[2].endswith("photo_3\n")


# ---------------- MiniCPM-V-4_5 ----------------

def chat_join(content):
    """Official MiniCPM-V chat(): images become (<image>./</image>), parts are joined with newlines."""
    return "\n".join("(<image>./</image>)" if q["type"] == "image" else q["text"] for q in content) if isinstance(content, list) else content


@pytest.mark.parametrize("kw", [{}, {"enable_thinking": True}, {"enable_thinking": False}])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_minicpm_text_only_identical(kw, n):
    off = render("official/minicpm_v_4_5.jinja", TEXT_CONV[:n], add_generation_prompt=True, **kw)
    assert render("minicpm_v_interleaved.jinja", as_list(TEXT_CONV[:n]), add_generation_prompt=True, **kw) == off
    assert render("minicpm_v_interleaved.jinja", TEXT_CONV[:n], add_generation_prompt=True, **kw) == off


def test_minicpm_assistant_reasoning_identical():
    conv = TEXT_CONV[:3] + [{"role": "user", "content": "next"}]
    conv[2] = {"role": "assistant", "content": "<think>\nhmm\n</think>\n\n{\"a\": 1}"}
    off = render("official/minicpm_v_4_5.jinja", conv, add_generation_prompt=True, enable_thinking=True)
    assert render("minicpm_v_interleaved.jinja", as_list(conv), add_generation_prompt=True, enable_thinking=True) == off


def test_minicpm_images_in_place():
    msgs = image_conv()
    new = render("minicpm_v_interleaved.jinja", msgs, add_generation_prompt=True, enable_thinking=True)
    off = render("official/minicpm_v_4_5.jinja", [{**m, "content": chat_join(m["content"])} for m in msgs], add_generation_prompt=True, enable_thinking=True)
    assert new == off                                                          # identical to the official chat() string rendered with the official template
    assert "START photo_111147\n\n(<image>./</image>)\n\nphoto_2\n\n(<image>./</image>)" in new
    assert new.count("(<image>./</image>)") == 3
    with pytest.raises(TypeError):                                             # the official template fails on a list (the HTTP 400 seen in the smoke test)
        render("official/minicpm_v_4_5.jinja", msgs, add_generation_prompt=True)


def test_official_copies_match_model_dirs():
    from actspace_eval.data import load_eval_config
    models = load_eval_config()["models"]
    for name, src, key in (("llava_onevision_7b.jinja", Path(models["LLaVA-OneVision-7B"]["path"]) / "chat_template.json", "chat_template"),
                           ("minicpm_v_4_5.jinja", Path(models["MiniCPM-V-4_5"]["path"]) / "tokenizer_config.json", "chat_template")):
        if not Path(src).exists():
            pytest.skip("model weights are not available locally")
        assert (T / "official" / name).read_text() == json.loads(Path(src).read_text())[key]


# ---------------- interleaving check and per-image token pairing ----------------

def call(rendered, content, t=0, group="arkit_g_000000"):
    imgs = [{"type": "image", "w": 640, "h": 480} if q == "I" else {"type": "text", "text": q} for q in content]
    return {"type": "call", "tag": {"t": t, "episode": group}, "group_id": group,
            "request": {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": imgs}]},
            "rendered_prompt": {"text": rendered}}


CONTENT = ["Photographs (2):\nSTART photo_111147\n", "I", "\nphoto_2\n", "I", "\nAnswer all questions."]


def test_order_interleaved_with_folded_digits():
    r = call("<|im_start|>user\nPhotographs (2):\nSTART photo_1×447\n<|vision_start|><|image_pad|>×300<|vision_end|>\nphoto_2\n"
             "<|vision_start|><|image_pad|>×300<|vision_end|>\nAnswer all questions.<|im_end|>", CONTENT)
    assert content_order.check_call(r) == "interleaved"


def test_order_images_first():
    r = call("<|im_start|>user <image>×1460\nPhotographs (2):\nSTART photo_111147\n\n\nphoto_2\n\n\nAnswer all questions.<|im_end|>", CONTENT)
    assert content_order.check_call(r) == "images_first"


def test_order_minicpm_slices_and_no_image():
    r = call("<|im_start|>user\nPhotographs (2):\nSTART photo_111147\n\n<image><unk>×64</image><slice><unk>×64</slice>\n\nphoto_2\n\n"
             "<image><unk>×64</image>\n\nAnswer all questions.<|im_end|>", CONTENT)
    assert content_order.check_call(r) == "interleaved"
    assert content_order.check_call(call("x", ["only text"])) == "no_image"
    assert content_order.check_call({**r, "rendered_prompt": {"error": "timeout"}}) == "no_render"
    assert content_order.order_ok(content_order.Counter(interleaved=3, no_render=1))
    assert not content_order.order_ok(content_order.Counter(interleaved=3, images_first=1))


def write_run(tmp_path, calls):
    d = tmp_path / "M" / "passive_8" / "smoke2_r0"
    d.mkdir(parents=True)
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return d


def test_tokens_merged_adjacent_placeholders(tmp_path):
    """Placeholders of adjacent images merged into one segment (<image>×1460 = 2 × 730): paired as 730 per image, no longer
    recorded as "unpaired"."""
    c = call("<|im_start|>user <image>×1460\nPhotographs (2):\nSTART photo_111147\n\n\nphoto_2\n\n\nAnswer all questions.<|im_end|>", CONTENT)
    d = write_run(tmp_path, [c])
    assert tokens_by_orientation(d) == {"arkit 640×480": [730]}


def test_tokens_merged_uneven_not_paired(tmp_path):
    c = call("<|im_start|>user <image>×1461\nPhotographs (2):\nSTART photo_111147\n\nAnswer all questions.<|im_end|>", CONTENT)
    assert tokens_by_orientation(write_run(tmp_path, [c])) == {"unpaired": ["1/2"]}


def test_tokens_merged_mixed_sizes_not_averaged(tmp_path):
    """A segment mixing landscape and portrait images does not count towards tokens per image (700 and 760 are not averaged
    into 730)."""
    c = call("<|im_start|>user <image>×1460\nPhotographs (2):\nSTART photo_111147\n\n\nphoto_2\n\n\nAnswer all questions.<|im_end|>", CONTENT)
    c["request"]["messages"][1]["content"][3] = {"type": "image", "w": 480, "h": 640}
    assert tokens_by_orientation(write_run(tmp_path, [c])) == {}


def test_gate_fails_on_order(tmp_path):
    c = call("<|im_start|>user <image>×1460\nPhotographs (2):\nSTART photo_111147\n\n\nphoto_2\n\n\nAnswer all questions.<|im_end|>", CONTENT)
    write_run(tmp_path, [c])
    g = gate_result(tmp_path, "M", "smoke2_r0", ["passive_8"], extra=True)
    assert not g["pass"] and any("interleaving order" in w for w in g["why"])
    assert g["checks"]["order"] == {"passive_8": {"images_first": 1}}
