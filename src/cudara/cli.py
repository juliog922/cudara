"""
Cudara CLI.
===========
Command-line interface for interacting with the Cudara inference server.
Provides commands for serving the API, managing models, and running queries.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Default configuration constants
DEFAULT_HOST: str = os.getenv("CUDARA_HOST", "http://localhost:8000")
CONFIG_FILE: Path = Path.home() / ".cudara" / "config.json"


def get_config() -> Dict[str, str]:
    """Load CLI configuration from disk."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"host": DEFAULT_HOST}


def save_config(config: Dict[str, str]) -> None:
    """Save CLI configuration to disk."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_client(host: Optional[str] = None, timeout: float = 600.0) -> httpx.Client:
    """Create an HTTP client configured for the Cudara API."""
    config = get_config()
    base_url = host or config.get("host", DEFAULT_HOST)
    return httpx.Client(base_url=base_url, timeout=timeout)


def print_error(msg: str) -> None:
    """Print an error message to stderr in red."""
    print(f"\033[31mError: {msg}\033[0m", file=sys.stderr)


def print_success(msg: str) -> None:
    """Print a success message to stdout in green."""
    print(f"\033[32m{msg}\033[0m")


def print_info(msg: str) -> None:
    """Print an info message to stdout in cyan."""
    print(f"\033[36m{msg}\033[0m")


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable strings like Ollama."""
    if size_bytes == 0:
        return "0 B"

    size = float(size_bytes)  # Add this line
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def extract_images_from_prompt(prompt: str) -> Tuple[str, List[str]]:
    """
    Parses the prompt for local image file paths, removes them from the text,
    and returns their base64 encoded strings for multimodal support.
    """
    images = []
    clean_prompt_parts = []
    for word in prompt.split():
        # Basic check to see if word is a file path to an image
        if Path(word).is_file() and word.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            try:
                with open(word, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode("utf-8"))
            except Exception as e:
                print_error(f"Failed to read image {word}: {e}")
        else:
            clean_prompt_parts.append(word)

    return " ".join(clean_prompt_parts), images


# --- CLI Commands ---


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the uvicorn server."""
    import uvicorn

    print_info(f"Starting Cudara server on {args.host}:{args.port}")
    uvicorn.run(
        "src.cudara.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


def cmd_list(args: argparse.Namespace) -> None:
    """List available models via the API (matches 'ollama ls')."""
    try:
        with get_client(args.host) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])

            if not models:
                print("No models available.")
                return

            print(f"{'NAME':<30} {'ID':<15} {'SIZE':<10} {'MODIFIED'}")
            for model in models:
                name = model.get("name", "unknown")
                digest = model.get("digest", "")[7:19]  # strip sha256: and take first 12 chars
                size = format_size(model.get("size", 0))
                # Simple datetime formatting
                mod_raw = model.get("modified_at", "").split("T")[0]
                print(f"{name:<30} {digest:<15} {size:<10} {mod_raw}")

    except Exception as e:
        print_error(f"Cannot connect to server: {str(e)}")
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

            print(f"{'NAME':<30} {'ID':<15} {'SIZE (VRAM)':<15} {'EXPIRES AT'}")
            for model in models:
                name = model.get("name", "unknown")
                digest = model.get("digest", "")[7:19]
                size = format_size(model.get("size_vram", 0))
                expires = model.get("expires_at", "").replace("T", " ")[:19]
                print(f"{name:<30} {digest:<15} {size:<15} {expires}")

    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_pull(args: argparse.Namespace) -> None:
    """Trigger a model pull and stream progress."""
    try:
        with get_client(args.host) as client:
            with client.stream("POST", "/api/pull", json={"model": args.model, "stream": True}) as response:
                if response.status_code == 400:
                    print_error(f"Model '{args.model}' is not in the allowed list (models.json).")
                    sys.exit(1)

                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        print(f"\rpulling... {status:<30}", end="", flush=True)
                print("\n\033[32msuccess\033[0m")

    except Exception as e:
        print("\n")
        print_error(str(e))
        sys.exit(1)


def cmd_rm(args: argparse.Namespace) -> None:
    """Remove a model."""
    try:
        with get_client(args.host) as client:
            response = client.request("DELETE", "/api/delete", json={"model": args.model})
            response.raise_for_status()
            print_success(f"deleted '{args.model}'")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop a running model by forcing its keep_alive to 0."""
    try:
        with get_client(args.host) as client:
            response = client.post("/api/generate", json={"model": args.model, "keep_alive": 0})
            response.raise_for_status()
            print_success(f"stopped '{args.model}'")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """
    Run a model. Acts as a single-prompt completion if a prompt is provided,
    otherwise drops into an interactive chat session (matches 'ollama run').
    """
    try:
        with get_client(args.host) as client:
            # First, verify the model exists/is ready
            response = client.post("/api/show", json={"model": args.model})
            if response.status_code == 404:
                print_error(f"Error: model '{args.model}' not found, try pulling it first")
                sys.exit(1)

            # 1. Single Prompt Execution
            if args.prompt or not sys.stdin.isatty():
                if args.prompt:
                    prompt = " ".join(args.prompt)
                else:
                    prompt = sys.stdin.read().strip()

                if not prompt:
                    return

                prompt, images = extract_images_from_prompt(prompt)

                payload: Dict[str, Any] = {
                    "model": args.model,
                    "prompt": prompt,
                    "stream": True,
                }
                if images:
                    payload["images"] = images

                with client.stream("POST", "/api/generate", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            print(data.get("response", ""), end="", flush=True)
                print()
                return

            # 2. Interactive Chat Execution
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
                if user_input.lower() in ["/bye", "exit", "quit"]:
                    break
                if user_input == "/clear":
                    messages = []
                    print_info("Cleared session context")
                    continue

                # Multiline support
                if user_input.startswith('"""'):
                    lines = [user_input[3:]]
                    while True:
                        line = input("... ")
                        if line.endswith('"""'):
                            lines.append(line[:-3])
                            break
                        lines.append(line)
                    user_input = "\n".join(lines).strip()

                user_input, images = extract_images_from_prompt(user_input)

                msg: Dict[str, Any] = {"role": "user", "content": user_input}
                if images:
                    msg["images"] = images
                messages.append(msg)

                assistant_msg = ""
                with client.stream(
                    "POST", "/api/chat", json={"model": args.model, "messages": messages, "stream": True}
                ) as stream_response:
                    stream_response.raise_for_status()
                    for line in stream_response.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")

                            # Optional: Handle separate thinking output if returned
                            thinking = data.get("thinking")
                            if thinking:
                                # Grey text for thinking
                                print(f"\033[90m{thinking}\033[0m", end="", flush=True)

                            print(chunk, end="", flush=True)
                            assistant_msg += chunk

                print("\n")
                messages.append({"role": "assistant", "content": assistant_msg})

    except Exception as e:
        print_error(str(e))
        sys.exit(1)


# --- Stubs for strict Ollama CLI compatibility ---


def cmd_create(args: argparse.Namespace) -> None:
    """Create a customized model (Stub)."""
    try:
        with get_client(args.host) as client:
            response = client.post("/api/create", json={"name": "stub", "modelfile": "stub"})
            if response.status_code == 501:
                print_error("The 'create' command is not yet implemented in Cudara.")
            else:
                print(response.json())
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_launch(args: argparse.Namespace) -> None:
    """Launch integrations (Stub)."""
    if getattr(args, "integration", None):
        print_info(f"Launching integration: {args.integration}")
    else:
        print_info("Interactive integration launcher not yet implemented.")
        print("Supported targets: OpenCode, Claude Code, Codex, Droid")


def cmd_signin(args: argparse.Namespace) -> None:
    """Sign in (Stub)."""
    print_info("Authentication is managed via Hugging Face token (HF_TOKEN env var).")


def cmd_signout(args: argparse.Namespace) -> None:
    """Sign out (Stub)."""
    print_info("Authentication is managed via Hugging Face token (HF_TOKEN env var).")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(prog="cudara", description="Cudara CLI")
    parser.add_argument("--version", action="version", version="cudara 1.0.0")
    subparsers = parser.add_subparsers(dest="command", title="commands")

    # Serve Command
    serve_parser = subparsers.add_parser("serve", help="Start the Cudara server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument("--workers", type=int, default=1)
    serve_parser.set_defaults(func=cmd_serve)

    # Run Command (Unified single-prompt and interactive chat)
    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument("model", help="Model name to run")
    run_parser.add_argument("prompt", nargs="*", help="Optional prompt to execute a single completion")
    run_parser.add_argument("--host")
    run_parser.set_defaults(func=cmd_run)

    # List/Ps Commands
    list_parser = subparsers.add_parser("list", aliases=["ls"], help="List models")
    list_parser.add_argument("--host")
    list_parser.set_defaults(func=cmd_list)

    ps_parser = subparsers.add_parser("ps", help="Show running models")
    ps_parser.add_argument("--host")
    ps_parser.set_defaults(func=cmd_ps)

    # Pull/Rm/Stop Commands
    pull_parser = subparsers.add_parser("pull", help="Pull a model from a registry")
    pull_parser.add_argument("model", help="Model name to pull")
    pull_parser.add_argument("--host")
    pull_parser.set_defaults(func=cmd_pull)

    rm_parser = subparsers.add_parser("rm", aliases=["delete"], help="Remove a model")
    rm_parser.add_argument("model", help="Model name to remove")
    rm_parser.add_argument("--host")
    rm_parser.set_defaults(func=cmd_rm)

    stop_parser = subparsers.add_parser("stop", help="Stop a running model")
    stop_parser.add_argument("model", help="Model name to stop")
    stop_parser.add_argument("--host")
    stop_parser.set_defaults(func=cmd_stop)

    # Integration and Stub Commands
    launch_parser = subparsers.add_parser("launch", help="Launch external integrations")
    launch_parser.add_argument("integration", nargs="?", help="Integration name")
    launch_parser.add_argument("--config", action="store_true")
    launch_parser.add_argument("--model")
    launch_parser.set_defaults(func=cmd_launch)

    create_parser = subparsers.add_parser("create", help="Create a model from a Modelfile")
    create_parser.add_argument("-f", "--file", help="Path to Modelfile")
    create_parser.set_defaults(func=cmd_create)

    signin_parser = subparsers.add_parser("signin", help="Sign in")
    signin_parser.set_defaults(func=cmd_signin)

    signout_parser = subparsers.add_parser("signout", help="Sign out")
    signout_parser.set_defaults(func=cmd_signout)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
