"""Central configuration for the friend-simulator chatbot.

Everything tunable lives here so the individual scripts stay simple.
Defaults are chosen for an RTX 2070 (8 GB VRAM, Turing -> no bf16) and 16 GB RAM.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
# Ollama only ships quantized GGUF files, which cannot be finetuned. We therefore
# pull the same model twice: the Ollama tag (for baseline testing + its chat
# template) and the original Hugging Face weights (for finetuning).
#
# Other combinations that fit 8 GB with QLoRA:
#   "qwen2.5:1.5b" / "Qwen/Qwen2.5-1.5B-Instruct"   (faster, weaker)
#   "qwen2.5:7b"   / "Qwen/Qwen2.5-7B-Instruct"     (tight; set MAX_SEQ_LEN=512, BATCH_SIZE=1)
OLLAMA_BASE_MODEL = "qwen2.5:3b"
HF_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Name of the finetuned model inside Ollama.
OLLAMA_FINETUNED_MODEL = "friends-bot"
# Quantization used by `ollama create` (q4_K_M is small and good; q8_0 is higher quality).
OLLAMA_QUANTIZATION = "q4_K_M"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODELS_DIR = ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "base"          # HF weights
ADAPTER_DIR = MODELS_DIR / "lora_adapter"     # finetuning output (LoRA weights only)
MERGED_MODEL_DIR = MODELS_DIR / "merged"      # base + LoRA merged, fp16 safetensors
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"

DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
PARTICIPANTS_FILE = DATA_DIR / "participants.json"

MODELFILE = ROOT / "Modelfile"

# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
# Your own sender name as it appears in the chat export. Used by chat.py so the
# bot hands the turn back to you whenever it predicts that *you* would speak next.
USER_NAME = "Julian"

# How many previous messages the model sees as context for each prediction.
CONTEXT_MESSAGES = 20
# Messages further apart than this start a new "conversation" (context is not
# carried across). Only used if your parser provides timestamps.
SESSION_GAP_HOURS = 6
# Skip training targets that are shorter than this (e.g. "ok", reactions).
MIN_TARGET_CHARS = 1
# Senders whose messages are never used as training targets (they still appear
# as context). E.g. ["Me"] if the bot should never learn to imitate you.
EXCLUDED_TARGET_SENDERS: list[str] = []
VAL_FRACTION = 0.03
SEED = 42

# --------------------------------------------------------------------------- #
# Finetuning (QLoRA)
# --------------------------------------------------------------------------- #
MAX_SEQ_LEN = 1024            # tokens per sample; oldest context is dropped to fit
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8          # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_STEPS = 20
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200

# --------------------------------------------------------------------------- #
# Inference (chat.py)
# --------------------------------------------------------------------------- #
OLLAMA_HOST = "http://localhost:11434"
TEMPERATURE = 0.8
TOP_P = 0.9
NUM_CTX = 2048
# Max number of friend messages generated in a row before you get the turn back.
MAX_BOT_TURNS = 4
