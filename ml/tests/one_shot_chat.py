"""Bypass the REPL and do a single-shot generation.

Loads the base model + your S1 adapter, renders the synthetic SAR patch,
sends one question, prints the reply. Used to prove the local stack
end-to-end without needing a real BigEarthNet tile on disk.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # ml/ on path so we can import serve, serve_utils

from serve import ChatContent, ChatMessage, _generate, _messages_to_qwen, load_model, resolve_base_model  # noqa: E402
from serve_utils import image_to_b64, load_image  # noqa: E402

ADAPTER = r"C:\Users\shubh\100 DAYS CODE\Qwen2.5foiiles"
PATCH   = HERE.parent / "data" / "smoke" / "s1_patch"
QUESTIONS = [
    "Which land-cover classes are present?",
    "Is there any built-up or urban area in this scene?",
    "Do you see any water bodies?",
    "Describe this remote sensing image in detail.",
]

def main() -> None:
    print(f"[step 1/4] rendering S1 patch from {PATCH}", flush=True)
    img, label = load_image(PATCH)
    print(f"           rendered: {label} ({img.size[0]}x{img.size[1]})", flush=True)

    print(f"[step 2/4] resolving base model for adapter {ADAPTER}", flush=True)
    base = resolve_base_model(ADAPTER, None)
    print(f"           base = {base}", flush=True)

    print(f"[step 3/4] loading base + adapter", flush=True)
    t0 = time.perf_counter()
    processor, model = load_model(base, ADAPTER)
    print(f"           loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    label_str = f"{base.split('/')[-1]} + RS-LoRA"
    image_b64 = image_to_b64(img, format="PNG")
    history = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[step 4.{i}/4] asking: {question!r}", flush=True)
        parts = [ChatContent(type="text", text=question)]
        if i == 1:  # attach the image only on turn 1
            parts.insert(0, ChatContent(type="image", imageB64=image_b64))
        history.append(ChatMessage(role="user", content=parts))

        qwen_msgs, images = _messages_to_qwen(history)
        t0 = time.perf_counter()
        answer, confidence = _generate(model, processor, qwen_msgs, images, 128, 0.0, label_str)
        dt = time.perf_counter() - t0

        history.append(ChatMessage(role="assistant", content=answer))
        print("-" * 70)
        print(f"you> {question}")
        print(f"bot> {answer}")
        print(f"     confidence {confidence:.2f} · {dt*1000:.0f} ms")
        print("-" * 70)


if __name__ == "__main__":
    main()
