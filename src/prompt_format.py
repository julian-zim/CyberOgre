"""Shared prompt format for training and inference.

Training and chatting MUST render prompts identically, so both import from here.

One model plays every friend: it sees the recent transcript and predicts the
next message *including who sends it*, as "Sender: text".
"""

SYSTEM_TEMPLATE = (
    "You are simulating a group chat between friends: {participants}. "
    "Given the recent chat transcript, write the next message in the chat, "
    "exactly in the style of whoever is most likely to send it. "
    "Answer in the format 'Name: message'."
)


def system_prompt(participants: list[str]) -> str:
    return SYSTEM_TEMPLATE.format(participants=", ".join(participants))


def format_line(sender: str, text: str) -> str:
    return f"{sender}: {text}"


def build_prompt(context: list[dict], participants: list[str]) -> list[dict]:
    """context: list of {"sender": ..., "text": ...}, oldest first."""
    transcript = "\n".join(format_line(m["sender"], m["text"]) for m in context)
    return [
        {"role": "system", "content": system_prompt(participants)},
        {"role": "user", "content": transcript},
    ]


def build_completion(sender: str, text: str) -> dict:
    return {"role": "assistant", "content": format_line(sender, text)}


def parse_reply(reply: str, participants: list[str]) -> tuple[str | None, str]:
    """Split a model reply "Name: text" into (sender, text).

    Returns sender=None if the reply doesn't start with a known participant.
    """
    reply = reply.strip()
    for name in sorted(participants, key=len, reverse=True):
        prefix = f"{name}:"
        if reply.startswith(prefix):
            return name, reply[len(prefix):].strip()
    return None, reply
