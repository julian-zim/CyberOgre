"""Fetch the base model.

  1. `ollama pull` the quantized model -> lets you chat with the untuned
     baseline and provides the chat template for the finetuned model later.
  2. Download the original Hugging Face weights -> these are what we finetune
     (Ollama's GGUF files are quantized and not trainable).

Usage:  python fetch_model.py [--skip-ollama] [--skip-hf]
"""
import argparse
import shutil
import subprocess
import sys

import config


def pull_ollama(tag: str) -> None:
    if shutil.which("ollama") is None:
        sys.exit("`ollama` not found on PATH. Install it from https://ollama.com first.")
    print(f"==> ollama pull {tag}")
    subprocess.run(["ollama", "pull", tag], check=True)


def download_hf(repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"==> Downloading {repo_id} -> {config.BASE_MODEL_DIR}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=config.BASE_MODEL_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-ollama", action="store_true", help="don't pull the Ollama model")
    parser.add_argument("--skip-hf", action="store_true", help="don't download the HF weights")
    args = parser.parse_args()

    if not args.skip_ollama:
        pull_ollama(config.OLLAMA_BASE_MODEL)
    if not args.skip_hf:
        download_hf(config.HF_BASE_MODEL)
    print("Done.")


if __name__ == "__main__":
    main()
