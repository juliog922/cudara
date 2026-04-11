"""
Cudara CLI.

Command-line interface for interacting with the Cudara inference server.
Provides commands for serving the API, managing models, and running queries.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Version — matches main.py and pyproject.toml
# ---------------------------------------------------------------------------
VERSION = "0.0.1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_HOST: str = os.getenv("CUDARA_HOST", "http://localhost:8000")
DEFAULT_TIMEOUT: float = float(os.getenv("CUDARA_TIMEOUT", "600"))
CONFIG_FILE: Path = Path.home() / ".cudara" / "config.json"


def get_config() -> Dict[str, str]:
    """Load CLI configuration from disk."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"host": DEFAULT_HOST}
    return {"host": DEFAULT_HOST}


def save_config(config: Dict[str, str]) -> None:
    """Save CLI configuration to disk."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as exc:
        print_error(f"Failed to save config: {exc}")


def get_client(host: Optional[str] = None, timeout: Optional[float] = None) -> httpx.Client:
    """Create an HTTP client configured for the Cudara API."""
    config = get_config()
    base_url = host or config.get("host", DEFAULT_HOST)
    t = timeout if timeout is not None else DEFAULT_TIMEOUT
    return httpx.Client(base_url=base_url, timeout=t)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def print_error(msg: str) -> None:
    """Print an error message to stderr in red."""
    print(f"\033[31mError: {msg}\033[0m", file=sys.stderr)


def print_success(msg: str) -> None:
    """Print a success message in green."""
    print(f"\033[32m{msg}\033[0m")


def print_info(msg: str) -> None:
    """Print an info message in cyan."""
    print(f"\033[36m{msg}\033[0m")


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable strings."""
    if size_bytes == 0:
        return "0 B"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def extract_images_from_prompt(prompt: str) -> Tuple[str, List[str]]:
    """Parse local image file paths from prompt and base64-encode them."""
    images: List[str] = []
    clean_parts: List[str] = []
    supported = (".png", ".jpg", ".jpeg", ".webp")

    for word in prompt.split():
        path = Path(word)
        if path.is_file() and word.lower().endswith(supported):
            try:
                with open(path, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode("utf-8"))
            except OSError as exc:
                print_error(f"Failed to read image {word}: {exc}")
        else:
            clean_parts.append(word)

    return " ".join(clean_parts), images


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> None:
    """Start the uvicorn server."""
    try:
        import uvicorn
    except ImportError:
        print_error("uvicorn is required to serve. Install with: pip install uvicorn")
        sys.exit(1)

    print_info(f"Starting Cudara server on {args.host}:{args.port}")
    uvicorn.run(
        "cudara.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


def cmd_list(args: argparse.Namespace) -> None:
    """List available models (matches 'ollama ls')."""
    try:
        with get_client(args.host) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])

            if not models:
                print("No models available.")
                return

            print(f"{'NAME':<40} {'ID':<15} {'SIZE':<10} {'MODIFIED'}")
            for model in models:
                name = model.get("name", "unknown")
                digest = model.get("digest", "")[7:19]
                size = format_size(model.get("size", 0))
                mod_raw = model.get("modified_at", "").split("T")[0]
                print(f"{name:<40} {digest:<15} {size:<10} {mod_raw}")

    except httpx.ConnectError:
        print_error(f"Cannot connect to server at {args.host or DEFAULT_HOST}. Is it running?")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print_error(f"Server returned {exc.response.status_code}: {exc.response.text}")
        sys.exit(1)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)


def cmd_ps(args: argparse.Namespace) -> None:
    """List running models (matches 'ollama ps')."""
    try:
        with get_client(args.host) as client:
            response = client.get("/api/ps")
            response.raise_for_status()
            models = response.json().get("models", [])

            if not models:
                print("No models currently running.")
                return

            print(f"{'NAME':<40} {'ID':<15} {'SIZE (VRAM)':<15} {'EXPIRES AT'}")
            for model in models:
                name = model.get("name", "unknown")
                digest = model.get("digest", "")[7:19]
                size = format_size(model.get("size_vram", 0))
                expires = model.get("expires_at", "").replace("T", " ")[:19]
                print(f"{name:<40} {digest:<15} {size:<15} {expires}")

    except httpx.ConnectError:
        print_error(f"Cannot connect to server at {args.host or DEFAULT_HOST}.")
        sys.exit(1)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)


def cmd_pull(args: argparse.Namespace) -> None:
    """Pull a model and stream progress."""
    try:
        with get_client(args.host, timeout=None) as client:
            with client.stream("POST", "/api/pull", json={"model": args.model, "stream": True}) as response:
                if response.status_code == 400:
                    # Read error body
                    body = ""
                    for chunk in response.iter_text():
                        body += chunk
                    try:
                        err = json.loads(body).get("error", body)
                    except json.JSONDecodeError:
                        err = body
                    print_error(err)
                    sys.exit(1)

                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        status = data.get("status", "")
                        if "error" in status.lower():
                            print(f"\n\033[31m{status}\033[0m")
                            sys.exit(1)
                        print(f"\rpulling... {status:<40}", end="", flush=True)
                print("\n\033[32msuccess\033[0m")

    except httpx.ConnectError:
        print_error(f"Cannot connect to server at {args.host or DEFAULT_HOST}.")
        sys.exit(1)
    except Exception as exc:
        print(file=sys.stderr)
        print_error(str(exc))
        sys.exit(1)


def cmd_rm(args: argparse.Namespace) -> None:
    """Remove a model."""
    try:
        with get_client(args.host) as client:
            response = client.request("DELETE", "/api/delete", json={"model": args.model})
            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found.")
                sys.exit(1)
            response.raise_for_status()
            print_success(f"deleted '{args.model}'")
    except httpx.ConnectError:
        print_error(f"Cannot connect to server at {args.host or DEFAULT_HOST}.")
        sys.exit(1)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop a running model (keep_alive=0)."""
    try:
        with get_client(args.host) as client:
            response = client.post("/api/generate", json={"model": args.model, "keep_alive": 0})
            response.raise_for_status()
            print_success(f"stopped '{args.model}'")
    except httpx.ConnectError:
        print_error(f"Cannot connect to server at {args.host or DEFAULT_HOST}.")
        sys.exit(1)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Run a model: single-prompt or interactive chat."""
    try:
        with get_client(args.host) as client:
            # Verify model exists
            response = client.post("/api/show", json={"model": args.model})
            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found. Try pulling it first.")
                sys.exit(1)

            # Single prompt mode
            if args.prompt or not sys.stdin.isatty():
                _run_single(client, args)
                return

            # Interactive chat mode
            _run_interactive(client, args)

    except httpx.ConnectError:
        print_error(f"Cannot connect to server at {args.host or DEFAULT_HOST}.")
        sys.exit(1)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)


def _run_single(client: httpx.Client, args: argparse.Namespace) -> None:
    """Execute a single prompt completion."""
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = sys.stdin.read().strip()

    if not prompt:
        return

    prompt, images = extract_images_from_prompt(prompt)
    payload: Dict[str, Any] = {"model": args.model, "prompt": prompt, "stream": True}
    if images:
        payload["images"] = images

    with client.stream("POST", "/api/generate", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                print(data.get("response", ""), end="", flush=True)
    print()


def _run_interactive(client: httpx.Client, args: argparse.Namespace) -> None:
    """Interactive chat session."""
    messages: List[Dict[str, Any]] = []
    print_info("Type '/bye' to exit, '/clear' to reset, '\"\"\"' for multiline input.")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("/bye", "exit", "quit"):
            break
        if user_input == "/clear":
            messages = []
            print_info("Cleared session context")
            continue

        # Multiline support
        if user_input.startswith('"""'):
            lines = [user_input[3:]]
            try:
                while True:
                    line = input("... ")
                    if line.endswith('"""'):
                        lines.append(line[:-3])
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            user_input = "\n".join(lines).strip()

        user_input, images = extract_images_from_prompt(user_input)

        msg: Dict[str, Any] = {"role": "user", "content": user_input}
        if images:
            msg["images"] = images
        messages.append(msg)

        assistant_msg = ""
        try:
            with client.stream(
                "POST",
                "/api/chat",
                json={"model": args.model, "messages": messages, "stream": True},
            ) as stream_response:
                stream_response.raise_for_status()
                for line in stream_response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        thinking = data.get("thinking")
                        if thinking:
                            print(f"\033[90m{thinking}\033[0m", end="", flush=True)

                        chunk = data.get("message", {}).get("content", "")
                        print(chunk, end="", flush=True)
                        assistant_msg += chunk
        except httpx.HTTPStatusError as exc:
            print_error(f"Server error: {exc.response.status_code}")
            continue

        print("\n")
        messages.append({"role": "assistant", "content": assistant_msg})


# ---------------------------------------------------------------------------
# Stub commands
# ---------------------------------------------------------------------------
def cmd_create(args: argparse.Namespace) -> None:
    """Create a customized model (stub)."""
    try:
        with get_client(args.host) as client:
            response = client.post("/api/create", json={"name": "stub", "modelfile": "stub"})
            if response.status_code == 501:
                print_error("The 'create' command is not yet implemented in Cudara.")
            else:
                print(response.json())
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)


def cmd_launch(args: argparse.Namespace) -> None:
    """Launch integrations (stub)."""
    if getattr(args, "integration", None):
        print_info(f"Launching integration: {args.integration}")
    else:
        print_info("Interactive integration launcher not yet implemented.")
        print("Supported targets: OpenCode, Claude Code, Codex, Droid")


def cmd_signin(args: argparse.Namespace) -> None:
    """Sign in (stub)."""
    print_info("Authentication is managed via HF_TOKEN environment variable.")


def cmd_signout(args: argparse.Namespace) -> None:
    """Sign out (stub)."""
    print_info("Authentication is managed via HF_TOKEN environment variable.")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="cudara", description="Cudara — Lightweight CUDA Inference Server")
    parser.add_argument("--version", action="version", version=f"cudara {VERSION}")
    subs = parser.add_subparsers(dest="command", title="commands")

    # Common args
    def add_host(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default=None, help="Server URL (default: $CUDARA_HOST or localhost:8000)")

    # serve
    sp = subs.add_parser("serve", help="Start the Cudara server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.add_argument("--workers", type=int, default=1)
    sp.set_defaults(func=cmd_serve)

    # run
    sp = subs.add_parser("run", help="Run a model")
    sp.add_argument("model", help="Model name")
    sp.add_argument("prompt", nargs="*", help="Optional prompt for single completion")
    add_host(sp)
    sp.set_defaults(func=cmd_run)

    # list
    sp = subs.add_parser("list", aliases=["ls"], help="List models")
    add_host(sp)
    sp.set_defaults(func=cmd_list)

    # ps
    sp = subs.add_parser("ps", help="Show running models")
    add_host(sp)
    sp.set_defaults(func=cmd_ps)

    # pull
    sp = subs.add_parser("pull", help="Pull a model")
    sp.add_argument("model", help="Model name")
    add_host(sp)
    sp.set_defaults(func=cmd_pull)

    # rm
    sp = subs.add_parser("rm", aliases=["delete"], help="Remove a model")
    sp.add_argument("model", help="Model name")
    add_host(sp)
    sp.set_defaults(func=cmd_rm)

    # stop
    sp = subs.add_parser("stop", help="Stop a running model")
    sp.add_argument("model", help="Model name")
    add_host(sp)
    sp.set_defaults(func=cmd_stop)

    # stubs
    sp = subs.add_parser("launch", help="Launch integrations")
    sp.add_argument("integration", nargs="?")
    sp.add_argument("--config", action="store_true")
    sp.add_argument("--model")
    sp.set_defaults(func=cmd_launch)

    sp = subs.add_parser("create", help="Create a model from Modelfile")
    sp.add_argument("-f", "--file", help="Path to Modelfile")
    add_host(sp)
    sp.set_defaults(func=cmd_create)

    sp = subs.add_parser("signin", help="Sign in")
    sp.set_defaults(func=cmd_signin)

    sp = subs.add_parser("signout", help="Sign out")
    sp.set_defaults(func=cmd_signout)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
