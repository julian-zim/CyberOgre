"""Turn parsed chat messages into training samples.

Each sample = up to CONTEXT_MESSAGES previous messages (context) + the next
message (target). Writes data/train.jsonl, data/val.jsonl, data/participants.json.

Usage:  python build_dataset.py path/to/export [more/exports ...]
"""
import argparse
import json
import random
from datetime import timedelta
from pathlib import Path

import config
from chat_parser import Message, parse_chat_history


def split_sessions(messages: list[Message]) -> list[list[Message]]:
    """Split into conversations wherever the time gap exceeds SESSION_GAP_HOURS."""
    gap = timedelta(hours=config.SESSION_GAP_HOURS)
    sessions: list[list[Message]] = []
    for msg in messages:
        prev = sessions[-1][-1] if sessions else None
        if prev is None or (msg.timestamp and prev.timestamp and msg.timestamp - prev.timestamp > gap):
            sessions.append([])
        sessions[-1].append(msg)
    return sessions


def make_samples(session: list[Message]) -> list[dict]:
    samples = []
    for i in range(1, len(session)):   # need at least one context message
        target = session[i]
        if target.sender in config.EXCLUDED_TARGET_SENDERS:
            continue
        if len(target.text.strip()) < config.MIN_TARGET_CHARS:
            continue
        context = session[max(0, i - config.CONTEXT_MESSAGES):i]
        samples.append({
            "context": [{"sender": m.sender, "text": m.text} for m in context],
            "response": {"sender": target.sender, "text": target.text},
        })
    return samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="chat export file(s)/dir(s) for chat_parser")
    args = parser.parse_args()

    samples: list[dict] = []
    participants: set[str] = set()
    for path in args.inputs:
        messages = parse_chat_history(path)
        participants.update(m.sender for m in messages)
        for session in split_sessions(messages):
            samples.extend(make_samples(session))
        print(f"{path}: {len(messages)} messages")

    if not samples:
        raise SystemExit("No training samples produced — check your parser output.")

    # Split by contiguous blocks rather than random samples: neighbouring samples
    # share most of their context, so a random split would leak into validation.
    n_val = max(1, int(len(samples) * config.VAL_FRACTION))
    val, train = samples[-n_val:], samples[:-n_val]
    random.Random(config.SEED).shuffle(train)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.TRAIN_FILE, train)
    write_jsonl(config.VAL_FILE, val)
    config.PARTICIPANTS_FILE.write_text(json.dumps(sorted(participants), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Participants: {sorted(participants)}")
    print(f"Wrote {len(train)} train / {len(val)} val samples to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
