"""Interactive chat REPL against the fine-tuned SatQuery VLM.

Two modes:

  # 1) Talk to an already-running ml/serve.py:
  python ml/serve.py --adapter path/to/ben-lora/adapter --port 8100
  python ml/chat.py  --image path/to/patch.tif  --server http://127.0.0.1:8100

  # 2) Boot the model in-process (no HTTP; handy for laptops):
  python ml/chat.py  --image path/to/patch.tif  --adapter path/to/ben-lora/adapter

Local files, folders, PNGs, JPGs and BigEarthNet-S1/S2 patch folders are all
accepted. Sentinel-1 VV/VH tiles are rendered client-side into the exact
pseudo-RGB the model saw during QLoRA training — see ml/serve_utils.py.

Slash commands inside the REPL:
  /image <path>   load a new scene (keeps chat history)
  /reset          drop all history (start fresh on the same image)
  /system <text>  override the system prompt for this session
  /save <path>    dump the transcript as JSONL
  /help           show these commands
  /quit           exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from serve_utils import (
    TRAIN_SYSTEM_PROMPT,
    image_to_b64,
    load_image,
)


# ---------------------------------------------------------------------------
# Terminal helpers (no external deps — Rich is optional and we don't require it)
# ---------------------------------------------------------------------------

class C:
    CYAN = "\x1b[36m"
    MAGENTA = "\x1b[35m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"


def _emit(prefix: str, colour: str, text: str) -> None:
    print(f"{colour}{prefix}{C.RESET} {text}")


def banner(model_label: str, image_label: str, backend: str) -> None:
    print()
    print(f"{C.BOLD}{C.CYAN}SatQuery chat{C.RESET}  ·  model: {C.YELLOW}{model_label}{C.RESET}")
    print(f"scene: {C.GREEN}{image_label}{C.RESET}")
    print(f"backend: {C.DIM}{backend}{C.RESET}")
    print(f"{C.DIM}type /help for commands, blank line to exit{C.RESET}")
    print()


# ---------------------------------------------------------------------------
# Backends: HTTP vs in-process
# ---------------------------------------------------------------------------

class HttpBackend:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.info = self._probe()

    def _probe(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            print(f"{C.RED}Cannot reach {self.base}: {exc}{C.RESET}", file=sys.stderr)
            print(f"{C.DIM}Start it with:  python ml/serve.py --adapter <path> --port 8100{C.RESET}", file=sys.stderr)
            sys.exit(2)

    @property
    def label(self) -> str:
        return self.info.get("model", "unknown")

    def chat(self, messages: list[dict], max_new_tokens: int, temperature: float) -> tuple[str, float, int]:
        body = json.dumps({
            "messages": messages,
            "maxNewTokens": max_new_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/v1/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["answer"], float(data.get("confidence", 0.0)), int(data.get("elapsedMs", 0))


class InProcessBackend:
    """Load the model in this Python process. Avoids the HTTP hop for demos."""

    def __init__(self, adapter: str | None, base_model_override: str | None):
        # Deferred imports — booting torch takes 5s+ and we don't want to pay
        # that when the user just wants HTTP mode.
        from serve import load_model, resolve_base_model

        base = resolve_base_model(adapter, base_model_override)
        self.processor, self.model = load_model(base, adapter)
        self.label = f"{base.split('/')[-1]}" + (" + RS-LoRA" if adapter else " (base)")

    def chat(self, messages: list[dict], max_new_tokens: int, temperature: float) -> tuple[str, float, int]:
        from serve import _generate, _messages_to_qwen
        from serve import ChatContent, ChatMessage  # pydantic models for parity

        typed = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                typed.append(ChatMessage(role=m["role"], content=content))
            else:
                typed.append(
                    ChatMessage(
                        role=m["role"],
                        content=[
                            ChatContent(
                                type=c["type"],
                                text=c.get("text"),
                                imageB64=c.get("imageB64"),
                            )
                            for c in content
                        ],
                    )
                )
        qwen_msgs, images = _messages_to_qwen(typed)
        started = time.perf_counter()
        answer, confidence = _generate(
            self.model, self.processor, qwen_msgs, images, max_new_tokens, temperature, self.label
        )
        return answer, confidence, int((time.perf_counter() - started) * 1000)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def build_user_message(text: str, image_b64: str | None) -> dict:
    parts: list[dict[str, Any]] = []
    if image_b64:
        parts.append({"type": "image", "imageB64": image_b64})
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def strip_images(messages: list[dict]) -> list[dict]:
    """Only send the image on the first user turn — cuts payload + latency and
    matches how the model was trained (single image per conversation)."""
    seen_image = False
    out: list[dict] = []
    for m in messages:
        if m["role"] != "user" or isinstance(m.get("content"), str):
            out.append(m)
            continue
        parts = []
        for part in m["content"]:
            if part["type"] == "image":
                if seen_image:
                    continue
                seen_image = True
            parts.append(part)
        out.append({"role": "user", "content": parts})
    return out


def save_transcript(path: Path, messages: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        # Strip base64 image blobs from the saved log so it stays small.
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = [
                    {"type": "image", "text": "<image omitted>"} if part.get("type") == "image" else part
                    for part in content
                ]
                m = {**m, "content": content}
            fh.write(json.dumps(m) + "\n")


def run_repl(backend, image_path: Path | None, system_prompt: str, max_new_tokens: int, temperature: float) -> None:
    history: list[dict] = [{"role": "system", "content": system_prompt}]

    def load_and_attach(path: Path) -> tuple[str, str]:
        pil, label = load_image(path)
        b64 = image_to_b64(pil, format="PNG")
        return b64, label

    current_image_b64: str | None = None
    current_image_label = "(no image)"
    if image_path is not None:
        current_image_b64, current_image_label = load_and_attach(image_path)

    banner_backend = getattr(backend, "base", "in-process")
    banner(backend.label, current_image_label, banner_backend)

    pending_image = current_image_b64  # first user turn gets the image

    while True:
        try:
            line = input(f"{C.BOLD}{C.MAGENTA}you>{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in ("/quit", "/exit", "/q"):
                break
            if cmd == "/help":
                print(__doc__.split("Slash commands", 1)[1])
                continue
            if cmd == "/reset":
                history = [{"role": "system", "content": system_prompt}]
                pending_image = current_image_b64
                _emit("[ok]", C.GREEN, "history cleared; image reattached")
                continue
            if cmd == "/image":
                if not arg:
                    _emit("[err]", C.RED, "usage: /image <path>")
                    continue
                try:
                    current_image_b64, current_image_label = load_and_attach(Path(arg))
                except Exception as exc:
                    _emit("[err]", C.RED, f"could not load image: {exc}")
                    continue
                pending_image = current_image_b64
                _emit("[ok]", C.GREEN, f"loaded {current_image_label}")
                continue
            if cmd == "/system":
                if not arg:
                    _emit("[err]", C.RED, "usage: /system <text>")
                    continue
                system_prompt = arg
                history = [{"role": "system", "content": system_prompt}]
                pending_image = current_image_b64
                _emit("[ok]", C.GREEN, "system prompt updated + history cleared")
                continue
            if cmd == "/save":
                target = Path(arg or f"chat-{int(time.time())}.jsonl")
                save_transcript(
                    target,
                    history,
                    {"model": backend.label, "image": current_image_label},
                )
                _emit("[ok]", C.GREEN, f"transcript -> {target}")
                continue
            _emit("[err]", C.RED, f"unknown command: {cmd} (try /help)")
            continue

        history.append(build_user_message(line, pending_image))
        pending_image = None  # only send image on turn 1

        try:
            payload = strip_images(history)
            answer, confidence, elapsed_ms = backend.chat(payload, max_new_tokens, temperature)
        except Exception as exc:
            _emit("[err]", C.RED, f"generation failed: {exc}")
            history.pop()  # rollback the user turn so retry works
            pending_image = current_image_b64
            continue

        history.append({"role": "assistant", "content": answer})
        _emit("bot>", C.CYAN + C.BOLD, answer)
        print(f"{C.DIM}  confidence {confidence:.2f} · {elapsed_ms} ms{C.RESET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=None, help="Path to a PNG/JPG/TIFF/BigEarthNet patch folder")
    parser.add_argument("--server", default=None, help="URL of a running ml/serve.py (e.g. http://127.0.0.1:8100)")
    parser.add_argument("--adapter", default=None, help="In-process mode: LoRA adapter dir")
    parser.add_argument("--model", default=None, help="In-process mode: override the base model")
    parser.add_argument("--system", default=TRAIN_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.server and args.adapter:
        parser.error("choose one: --server (HTTP) or --adapter (in-process)")

    if args.server:
        backend = HttpBackend(args.server)
    else:
        if not args.adapter:
            parser.error("need either --server URL or --adapter DIR")
        backend = InProcessBackend(args.adapter, args.model)

    run_repl(backend, args.image, args.system, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
