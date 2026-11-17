# Friends chatbot

Finetunes a small LLM on your group chat so it imitates your friends, then runs it in Ollama.

## Why not finetune the Ollama model directly?

Ollama stores models as quantized GGUF files, which can't be trained. So the pipeline is:

```
fetch_model.py       ollama pull qwen2.5:3b  +  download HF weights (Qwen/Qwen2.5-3B-Instruct)
build_dataset.py     your chat export -> data/train.jsonl, val.jsonl
finetune.py          QLoRA on the HF weights (4-bit base + LoRA adapters, fits 8 GB VRAM)
export_to_ollama.py  merge LoRA -> safetensors -> `ollama create friends-bot` (quantized to q4_K_M)
chat.py              chat with friends-bot through the Ollama API
```

## Setup

Use Python 3.11–3.12 (3.14 may not have wheels for bitsandbytes and friends yet):

```bash
pyenv install 3.12
cd chatbot
~/.pyenv/versions/3.12.*/bin/python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Make sure the Ollama server is running (`ollama serve`, or the system service/snap).

## Usage

1. Set `USER_NAME` in `src/config.py` to your own name as it appears in the export.
2. Implement `parse_chat_history()` in `src/chat_parser.py` for your export format.
3. Run:

```bash
python fetch_model.py
python build_dataset.py ../signal_chats/<your-export>
python finetune.py               # add --resume to continue from a checkpoint
python export_to_ollama.py
python chat.py                   # or: python chat.py --model qwen2.5:3b  (untuned baseline)
```

## How it works

A single model plays every friend. Each training sample is the last `CONTEXT_MESSAGES`
messages as a transcript (`Name: text` per line) and the target is the next message, again
as `Name: text`. That way the model learns *who* would reply as well as *what* they'd say.
In `src/chat.py`, when the model predicts that you (`USER_NAME`) would speak next, it hands the
turn back to you. Loss is computed only on the target message.

The prompt format lives in `src/prompt_format.py` and is shared by training and inference. If
you change it, retrain.

## Tuning for 8 GB VRAM / RTX 2070

- Turing has no bf16, so training runs in fp16 (handled automatically).
- Out of memory: lower `MAX_SEQ_LEN` (e.g. 512) or `BATCH_SIZE` to 1 and raise `GRAD_ACCUM_STEPS`.
- Faster or smaller: `qwen2.5:1.5b` / `Qwen/Qwen2.5-1.5B-Instruct`. Better but tight: the 7B
  variant with `MAX_SEQ_LEN=512`, `BATCH_SIZE=1`. Merging a 7B model needs ~15 GB RAM, which is
  more than you can spare with 16 GB.
- Rough speed: a 3B model with ~20k samples takes a few hours per epoch on a 2070.
- If the bot just copies phrases word for word, it's overfitting: use fewer epochs or a lower
  learning rate. If it sounds generic, train longer.

## Privacy

`data/`, `models/` and `outputs/` hold your chat contents (the finetuned weights memorize
some of it). They're gitignored. Don't share or upload them.
