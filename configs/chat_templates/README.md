# Chat templates

Every model receives the same interleaved content: each photo is placed between the text that precedes and follows it (frame
label, pose line), as in the request. vLLM passes message content to the chat template as a list of parts (OpenAI format). For three
model families, the official chat template either does not accept list content or moves all images in front of the text. For these
models, the evaluation serves a format-adapted template (`chat_template` in `configs/eval.yaml`). Each adapted template changes only
how the content is passed: text-only messages render exactly as with the official template, and image placeholders stay at their
original positions. `env.json` of every run records the SHA-256 of the template that was served (`chat_template_served.sha256`).

| File | Models | Official template | Difference from the official template | SHA-256 recorded in the paper runs |
|---|---|---|---|---|
| `sensenova_internvl3.jinja` | SenseNova-SI-1.1-InternVL3-2B, SenseNova-SI-1.2-InternVL3-8B, SenseNova-SI-1.3-InternVL3-8B | `chat_template` in the model's `tokenizer_config.json` | The official template concatenates the system message as a string, but vLLM passes it as a list. This template also accepts a list for the system message (it concatenates the text parts). Everything else is identical, including the default system prompt (InternVL's official system prompt, in Chinese) and the handling of user messages. | `deb38a23ab77e4a1fb70efb9cdb42742d9120251e5335005943e7b15eed9f274` |
| `llava_onevision_interleaved.jinja` | LLaVA-OneVision-7B | `official/llava_onevision_7b.jinja` (`chat_template.json` of the model) | The official template emits all images first, then all videos, then all text. This template emits images, videos and text in the original content order. The placeholders, the newline before each text part and the generation markers of assistant text are identical; only the order changes. | `d27d8cfc38f09181adba47871c856725efe099455cf240109b28459cc81a7578` |
| `minicpm_v_interleaved.jinja` | MiniCPM-V-4_5 | `official/minicpm_v_4_5.jinja` (`tokenizer_config.json` of the model) | The official template accepts only string content (a list raises an error). The macro `render_content` turns the list into the string that the model's official `chat()` builds: images become vLLM's placeholder `(<image>./</image>)`, parts are joined with newlines in their original order, and strings are returned unchanged. The rest of the template is verbatim; only `message.content` is replaced by `render_content(message.content)`. | `089a01cc883d6ee34aea77eddea2095d21437a3a2dcd22dc4cf9f9f5c531863f` |

- `official/` holds verbatim copies of the official templates. `tests/eval/test_templates.py::test_official_copies_match_model_dirs`
  compares them with the model directories when the weights are available locally.
- The three adapted files are kept **byte-identical** to the files used for the paper runs, so their hashes match the `env.json`
  records above. This is why two of them still contain short comments in Chinese. The comments only restate the differences
  described in the table and are not part of the rendered output.
- `tests/eval/test_templates.py` (offline, with jinja2) and `scripts/check_chat_templates.py` (against recorded requests, in the
  vLLM environment) check the equivalences stated in the table.
- Check the hashes with `sha256sum configs/chat_templates/*.jinja`.
