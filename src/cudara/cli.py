"""
Cudara CLI
==========
Command-line interface for Cudara inference server.

Usage:
    cudara serve                    # Start the server
    cudara list                     # List models
    cudara pull <model>             # Download a model
    cudara rm <model>               # Delete a model
    cudara run <model> [prompt]     # Run inference
    cudara chat <model>             # Interactive chat
    cudara ps                       # Show running model
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import httpx

DEFAULT_HOST = "http://localhost:8000"
CONFIG_FILE = Path.home() / ".cudara" / "config.json"


def get_config() -> dict:
    """
    Load CLI configuration from disk.

    Returns
    -------
    dict
        Configuration dictionary containing defaults or user settings.
    """
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"host": DEFAULT_HOST}


def save_config(config: dict):
    """
    Save CLI configuration to disk.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    """
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_client(host: Optional[str] = None, timeout: float = 300.0) -> httpx.Client:
    """
    Get a configured HTTP client.

    Parameters
    ----------
    host : str, optional
        Target host URL. If None, uses value from config.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    httpx.Client
        Ready-to-use HTTP client.
    """
    config = get_config()
    base_url = host or config.get("host", DEFAULT_HOST)
    return httpx.Client(base_url=base_url, timeout=timeout)


def print_error(msg: str):
    """Print error message to stderr in red."""
    print(f"\033[31mError: {msg}\033[0m", file=sys.stderr)


def print_success(msg: str):
    """Print success message to stdout in green."""
    print(f"\033[32m{msg}\033[0m")


def print_info(msg: str):
    """Print info message to stdout in cyan."""
    print(f"\033[36m{msg}\033[0m")


# =============================================================================
# Commands
# =============================================================================


def cmd_serve(args):
    """
    Start the Cudara server using Uvicorn.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing host, port, reload, workers.
    """
    import uvicorn

    print_info(f"Starting Cudara server on {args.host}:{args.port}")
    uvicorn.run(
        "src.cudara.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


def cmd_list(args):
    """
    List available models from the server.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """
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
                    "ready": "\033[32m",  # Green
                    "downloading": "\033[33m",  # Yellow
                    "quantizing": "\033[33m",  # Yellow
                    "error": "\033[31m",  # Red
                }.get(status, "\033[90m")

                print(f"{name:<50} {status_color}{status:<15}\033[0m {desc}")

    except httpx.ConnectError:
        print_error("Cannot connect to server. Is Cudara running?")
        print_info("Start with: cudara serve")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_pull(args):
    """
    Trigger a model download/pull on the server.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing model name.
    """
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

    except httpx.ConnectError:
        print_error("Cannot connect to server")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_rm(args):
    """
    Remove/delete a model from the server registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing model name.
    """
    try:
        with get_client(args.host) as client:
            response = client.request("DELETE", "/api/delete", json={"name": args.model})
            response.raise_for_status()
            print_success(f"Deleted {args.model}")

    except httpx.ConnectError:
        print_error("Cannot connect to server")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_run(args):
    """
    Run single-shot inference.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing model name, prompt, system prompt.
    """
    try:
        with get_client(args.host, timeout=600.0) as client:
            # Get prompt from args or stdin
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

            payload = {
                "model": args.model,
                "prompt": prompt,
                "stream": False,
            }

            if args.system:
                payload["system"] = args.system

            response = client.post("/api/generate", json=payload)

            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found or not ready")
                print_info("Use 'cudara pull <model>' to download")
                sys.exit(1)

            response.raise_for_status()
            data = response.json()

            print(data.get("response", ""))

            if args.verbose:
                duration_ms = data.get("total_duration", 0) / 1_000_000
                tokens = data.get("eval_count", 0)
                print_info(f"\n[{tokens} tokens, {duration_ms:.0f}ms]")

    except httpx.ConnectError:
        print_error("Cannot connect to server")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_chat(args):
    """
    Start an interactive chat session.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing model name and optional system prompt.
    """
    try:
        with get_client(args.host, timeout=600.0) as client:
            # Check model is available
            response = client.post("/api/show", json={"name": args.model})
            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found")
                sys.exit(1)

            print_info(f"Chatting with {args.model}")
            print_info("Type 'exit' or Ctrl+C to quit, '/clear' to reset\n")

            messages = []

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
                    json={
                        "model": args.model,
                        "messages": messages,
                        "stream": False,
                    },
                )

                response.raise_for_status()
                data = response.json()

                assistant_msg = data.get("message", {}).get("content", "")
                messages.append({"role": "assistant", "content": assistant_msg})

                print(f"\n{assistant_msg}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except httpx.ConnectError:
        print_error("Cannot connect to server")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_ps(args):
    """
    Show server status and active loaded model.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """
    try:
        with get_client(args.host) as client:
            response = client.get("/health")
            response.raise_for_status()
            data = response.json()

            print(f"Status:       {data.get('status', 'unknown')}")
            print(f"Active Model: {data.get('active_model') or 'none'}")
            print(f"CUDA:         {data.get('cuda_available', False)}")
            print(f"VRAM Used:    {data.get('vram_used', 'N/A')}")

    except httpx.ConnectError:
        print_error("Server is not running")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_show(args):
    """
    Show details for a specific model.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing model name.
    """
    try:
        with get_client(args.host) as client:
            response = client.post("/api/show", json={"name": args.model})

            if response.status_code == 404:
                print_error(f"Model '{args.model}' not found")
                sys.exit(1)

            response.raise_for_status()
            data = response.json()

            print(json.dumps(data, indent=2))

    except httpx.ConnectError:
        print_error("Cannot connect to server")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


def cmd_config(args):
    """
    Configure CLI settings like default host.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing new host value.
    """
    config = get_config()

    if args.host:
        config["host"] = args.host
        save_config(config)
        print_success(f"Set host to {args.host}")
    else:
        print(json.dumps(config, indent=2))


# =============================================================================
# Main
# =============================================================================


def main():
    """Main CLI entry point parsing arguments and dispatching commands."""
    parser = argparse.ArgumentParser(
        prog="cudara",
        description="Cudara - Lightweight CUDA Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cudara serve                     Start the server
  cudara list                      List available models
  cudara pull Qwen/Qwen2.5-3B      Download a model
  cudara run Qwen/Qwen2.5-3B "Hi"  Run inference
  cudara chat Qwen/Qwen2.5-3B      Interactive chat
  cudara ps                        Show server status
        """,
    )

    parser.add_argument("--version", action="version", version="cudara 1.0.0")

    subparsers = parser.add_subparsers(dest="command", title="commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the Cudara server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    serve_parser.set_defaults(func=cmd_serve)

    # list
    list_parser = subparsers.add_parser("list", aliases=["ls"], help="List models")
    list_parser.add_argument("--host", help="Server URL")
    list_parser.set_defaults(func=cmd_list)

    # pull
    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", help="Model ID")
    pull_parser.add_argument("--host", help="Server URL")
    pull_parser.set_defaults(func=cmd_pull)

    # rm
    rm_parser = subparsers.add_parser("rm", aliases=["delete"], help="Delete a model")
    rm_parser.add_argument("model", help="Model ID")
    rm_parser.add_argument("--host", help="Server URL")
    rm_parser.set_defaults(func=cmd_rm)

    # run
    run_parser = subparsers.add_parser("run", help="Run inference")
    run_parser.add_argument("model", help="Model ID")
    run_parser.add_argument("prompt", nargs="*", help="Prompt text")
    run_parser.add_argument("--system", "-s", help="System prompt")
    run_parser.add_argument("--host", help="Server URL")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Show stats")
    run_parser.set_defaults(func=cmd_run)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("model", help="Model ID")
    chat_parser.add_argument("--system", "-s", help="System prompt")
    chat_parser.add_argument("--host", help="Server URL")
    chat_parser.set_defaults(func=cmd_chat)

    # ps
    ps_parser = subparsers.add_parser("ps", help="Show server status")
    ps_parser.add_argument("--host", help="Server URL")
    ps_parser.set_defaults(func=cmd_ps)

    # show
    show_parser = subparsers.add_parser("show", help="Show model details")
    show_parser.add_argument("model", help="Model ID")
    show_parser.add_argument("--host", help="Server URL")
    show_parser.set_defaults(func=cmd_show)

    # config
    config_parser = subparsers.add_parser("config", help="Configure CLI")
    config_parser.add_argument("--host", help="Set default server URL")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
