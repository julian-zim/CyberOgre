"""Merge the LoRA adapter into the base model and register it with Ollama.

Steps:
  1. Load base model in fp16 on CPU (~6 GB RAM for 3B), apply LoRA, merge.
  2. Save merged weights as safetensors.
  3. Write a Modelfile reusing the chat template of the Ollama base model.
  4. `ollama create` — Ollama converts + quantizes the safetensors itself.

Usage:  python export_to_ollama.py [--skip-merge]
"""
import argparse
import subprocess

import config


def merge() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("==> Merging LoRA adapter into base model (CPU)")
    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_DIR, dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(base, config.ADAPTER_DIR).merge_and_unload()
    model.save_pretrained(config.MERGED_MODEL_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(config.BASE_MODEL_DIR).save_pretrained(config.MERGED_MODEL_DIR)
    print(f"Merged model saved to {config.MERGED_MODEL_DIR}")


def base_template() -> str:
    """Chat template of the Ollama base model, so prompting matches training."""
    out = subprocess.run(["ollama", "show", config.OLLAMA_BASE_MODEL, "--template"],
						 check=True, capture_output=True, text=True)
    return out.stdout.strip()


def write_modelfile() -> None:
    template = base_template()
    config.MODELFILE.write_text(
        f"FROM {config.MERGED_MODEL_DIR}\n"
        f'TEMPLATE """{template}"""\n'
        f'PARAMETER stop "<|im_end|>"\n'
        f'PARAMETER stop "<|endoftext|>"\n'
        f"PARAMETER temperature {config.TEMPERATURE}\n"
        f"PARAMETER top_p {config.TOP_P}\n"
        f"PARAMETER num_ctx {config.NUM_CTX}\n",
        encoding="utf-8",
    )
    print(f"Wrote {config.MODELFILE}")


def create_ollama_model() -> None:
    cmd = ["ollama", "create", config.OLLAMA_FINETUNED_MODEL, "-f", str(config.MODELFILE),
           "--quantize", config.OLLAMA_QUANTIZATION]
    print("==> " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done. Try it:  ollama run {config.OLLAMA_FINETUNED_MODEL}  or  python chat.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-merge", action="store_true", help="reuse an existing merged model")
    args = parser.parse_args()

    if not args.skip_merge:
        merge()
    write_modelfile()
    create_ollama_model()


if __name__ == "__main__":
    main()
