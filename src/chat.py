"""Chat with your simulated friends via Ollama.

You type as USER_NAME; after each of your messages the model writes up to
MAX_BOT_TURNS friend messages, stopping early when it predicts you'd speak next.

  <enter> on an empty line   let the friends keep talking
  /reset                     clear the conversation
  /quit                      exit

Usage:  python chat.py [--model friends-bot]   (use --model qwen2.5:3b to compare with the baseline)
"""
import argparse
import json

import requests

import config
from prompt_format import build_prompt, parse_reply


def generate(model: str, history: list[dict], participants: list[str]) -> str:
    messages = build_prompt(history[-config.CONTEXT_MESSAGES:], participants)
    resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", json={
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE, "top_p": config.TOP_P, "num_ctx": config.NUM_CTX},
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def friends_turn(model: str, history: list[dict], participants: list[str]) -> None:
    for _ in range(config.MAX_BOT_TURNS):
        sender, text = parse_reply(generate(model, history, participants), participants)
        if sender is None or sender == config.USER_NAME or not text:
            return   # model thinks it's your turn (or produced garbage)
        print(f"{sender}: {text}")
        history.append({"sender": sender, "text": text})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.OLLAMA_FINETUNED_MODEL)
    args = parser.parse_args()

    participants = json.loads(config.PARTICIPANTS_FILE.read_text(encoding="utf-8"))
    if config.USER_NAME not in participants:
        participants.append(config.USER_NAME)
    print(f"Chatting with {', '.join(p for p in participants if p != config.USER_NAME)} "
          f"as {config.USER_NAME}. /quit to exit.\n")

    history: list[dict] = []
    while True:
        try:
            line = input(f"{config.USER_NAME}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "/quit":
            break
        if line == "/reset":
            history.clear()
            print("-- conversation cleared --")
            continue
        if line:
            history.append({"sender": config.USER_NAME, "text": line})
        friends_turn(args.model, history, participants)


if __name__ == "__main__":
    main()
