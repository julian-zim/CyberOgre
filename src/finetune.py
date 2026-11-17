"""QLoRA finetuning of the base model on the chat dataset.

The base model is loaded in 4-bit (NF4) and only small LoRA adapter matrices
are trained, which keeps a 3B model well within 8 GB VRAM. Loss is computed on
the predicted message only, not on the context.

Usage:  python finetune.py [--resume]
"""
import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import config
from prompt_format import build_completion, build_prompt


def make_encoder(tokenizer, participants: list[str]):
    def tokenize(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(sample: dict) -> dict:
        context = sample["context"]
        completion = build_completion(sample["response"]["sender"], sample["response"]["text"])
        while True:
            prompt_msgs = build_prompt(context, participants)
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(prompt_msgs + [completion], tokenize=False)
            prompt_ids = tokenize(prompt_text)
            completion_ids = tokenize(full_text[len(prompt_text):])
            if len(prompt_ids) + len(completion_ids) <= config.MAX_SEQ_LEN or len(context) <= 1:
                break
            context = context[1:]   # drop the oldest message and retry

        input_ids = (prompt_ids + completion_ids)[: config.MAX_SEQ_LEN]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: config.MAX_SEQ_LEN]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — install a CUDA build of PyTorch (see README).")

    participants = json.loads(config.PARTICIPANTS_FILE.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # RTX 20xx (Turing) has no bf16 support -> compute in fp16.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_DIR,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=compute_dtype,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files={"train": str(config.TRAIN_FILE), "validation": str(config.VAL_FILE)})
    dataset = raw.map(make_encoder(tokenizer, participants), remove_columns=raw["train"].column_names)

    training_args = TrainingArguments(
        output_dir=str(config.CHECKPOINT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_steps=config.WARMUP_STEPS,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)
    print(f"LoRA adapter saved to {config.ADAPTER_DIR}")


if __name__ == "__main__":
    main()
