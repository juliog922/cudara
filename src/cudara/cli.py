"""
Cudara CLI.
===========
Command-line interface for interacting with the Cudara inference server.
Provides commands for serving the API, managing models, and running queries.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

# Default configuration constants
DEFAULT_HOST: str = "http://localhost:8000"
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


def get_client(host: Optional[str] = None, timeout: float = 300.0) -> httpx.Client:
    """
    Create an HTTP client configured for the Cudara API.

    Args:
        host: Optional host override (e.g., http://localhost:8000).
        timeout: Request timeout in seconds.

    Returns:
        httpx.Client: Configured client instance.
    """
    config = get_config()
    base_url = host or config.get("host", DEFAULT_HOST)
    return httpx.Client(base_url=base_url, timeout=timeout)


def print_error(msg: str) -> None:
    """Print an error message in red to stderr."""
    print(f"\033[31mError: {msg}\033[0m", file=sys.stderr)


def print_success(msg: str) -> None:
    """Print a success message in green."""
    print(f"\033[32m{msg}\033[0m")


def print_info(msg: str) -> None:
    """Print an informational message in cyan."""
    print(f"\033[36m{msg}\033[0m")


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
    """List available models via the API."""
    try:
        with get_client(args.host) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            if not models:
                print(
                    "No models available. Add models to models.json and use 'cudara pull <model>'"
                )
                return

            print(f"{'NAME':<50} {'STATUS':<15} {'DESCRIPTION'}")
            print("-" * 100)

            for model in models:
                name = model.get("name", "unknown")
                status = model.get("status", "unknown")
                desc = model.get("description", "")[:40]

                status_color = {
                    "ready": "\033[32m",
                    "downloading": "\033[33m",
                    "error": "\033[31m",
                }.get(status, "\033[90m")

                print(f"{name:<50} {status_color}{status:<15}\033[0m {desc}")

    except httpx.ConnectError:
        print_error("Cannot connect to server. Is Cudara running?")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_pull(args: argparse.Namespace) -> None:
    """Trigger a model pull."""
    try:
        with get_client(args.host) as client:
            print_info(f"Pulling {args.model}...")
            response = client.post("/api/pull", json={"name": args.model})

            if response.status_code == 403:
                print_error(f"Model '{args.model}' is not in the allowed list.")
                print_info("Add it to models.json first.")
                sys.exit(1)

            response.raise_for_status()
            print_success(f"Started downloading {args.model}")
            print_info("Use 'cudara list' to check progress")

    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_rm(args: argparse.Namespace) -> None:
    """Remove a model."""
    try:
        with get_client(args.host) as client:
            response = client.request("DELETE", "/api/delete", json={"name": args.model})
            response.raise_for_status()
            print_success(f"Deleted {args.model}")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single completion prompt."""
    try:
        with get_client(args.host, timeout=600.0) as client:
            if args.prompt:
                prompt = " ".join(args.prompt)
            elif not sys.stdin.isatty():
                prompt = sys.stdin.read().strip()
            else:
                print("Enter prompt (Ctrl+D to submit):")
                prompt = sys.stdin.read().strip()

            if not prompt:
                print_error("No prompt provided")
                sys.exit(1)

            payload: Dict[str, Any] = {
                "model": args.model,
                "prompt": prompt,
                "stream": False,
            }
            if args.system:
                payload["system"] = args.system

            response = client.post("/api/generate", json=payload)
            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found or not ready")
                sys.exit(1)

            response.raise_for_status()
            data = response.json()
            print(data.get("response", ""))

            if args.verbose:
                duration_ms = data.get("total_duration", 0) / 1_000_000
                tokens = data.get("eval_count", 0)
                print_info(f"\n[{tokens} tokens, {duration_ms:.0f}ms]")

    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session."""
    try:
        with get_client(args.host, timeout=600.0) as client:
            response = client.post("/api/show", json={"name": args.model})
            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found")
                sys.exit(1)

            print_info(f"Chatting with {args.model}")
            print_info("Type 'exit' or Ctrl+C to quit, '/clear' to reset\n")

            messages: list[Dict[str, str]] = []
            if args.system:
                messages.append({"role": "system", "content": args.system})

            while True:
                try:
                    user_input = input("\033[32m>>> \033[0m").strip()
                except EOFError:
                    break
                if not user_input:
                    continue
                if user_input.lower() == "exit":
                    break
                if user_input == "/clear":
                    messages = []
                    if args.system:
                        messages.append({"role": "system", "content": args.system})
                    print_info("Chat cleared")
                    continue

                messages.append({"role": "user", "content": user_input})
                response = client.post(
                    "/api/chat",
                    json={"model": args.model, "messages": messages, "stream": False},
                )
                response.raise_for_status()
                data = response.json()
                assistant_msg = data.get("message", {}).get("content", "")
                messages.append({"role": "assistant", "content": assistant_msg})
                print(f"\n{assistant_msg}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_ps(args: argparse.Namespace) -> None:
    """Show server status and active model."""
    try:
        with get_client(args.host) as client:
            response = client.get("/health")
            response.raise_for_status()
            data = response.json()
            print(f"Status:       {data.get('status', 'unknown')}")
            print(f"Active Model: {data.get('active_model') or 'none'}")
            print(f"CUDA:         {data.get('cuda_available', False)}")
            print(f"VRAM Used:    {data.get('vram_used', 'N/A')}")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_show(args: argparse.Namespace) -> None:
    """Show model details."""
    try:
        with get_client(args.host) as client:
            response = client.post("/api/show", json={"name": args.model})
            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found")
                sys.exit(1)
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_config(args: argparse.Namespace) -> None:
    """Read or write CLI configuration."""
    config = get_config()
    if args.host:
        config["host"] = args.host
        save_config(config)
        print_success(f"Set host to {args.host}")
    else:
        print(json.dumps(config, indent=2))


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(prog="cudara", description="Cudara CLI")
    parser.add_argument("--version", action="version", version="cudara 1.0.0")
    subparsers = parser.add_subparsers(dest="command", title="commands")

    serve_parser = subparsers.add_parser("serve", help="Start the Cudara server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument("--workers", type=int, default=1)
    serve_parser.set_defaults(func=cmd_serve)

    list_parser = subparsers.add_parser("list", aliases=["ls"])
    list_parser.add_argument("--host")
    list_parser.set_defaults(func=cmd_list)

    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("model")
    pull_parser.add_argument("--host")
    pull_parser.set_defaults(func=cmd_pull)

    rm_parser = subparsers.add_parser("rm", aliases=["delete"])
    rm_parser.add_argument("model")
    rm_parser.add_argument("--host")
    rm_parser.set_defaults(func=cmd_rm)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("model")
    run_parser.add_argument("prompt", nargs="*")
    run_parser.add_argument("--system", "-s")
    run_parser.add_argument("--host")
    run_parser.add_argument("--verbose", "-v", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("model")
    chat_parser.add_argument("--system", "-s")
    chat_parser.add_argument("--host")
    chat_parser.set_defaults(func=cmd_chat)

    ps_parser = subparsers.add_parser("ps")
    ps_parser.add_argument("--host")
    ps_parser.set_defaults(func=cmd_ps)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("model")
    show_parser.add_argument("--host")
    show_parser.set_defaults(func=cmd_show)

    config_parser = subparsers.add_parser("config")
    config_parser.add_argument("--host")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
