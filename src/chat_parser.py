"""Chat-history parsing — TO BE IMPLEMENTED BY YOU.

The rest of the pipeline only needs `parse_chat_history()` to return a
chronologically sorted list of `Message` objects. Everything about your export
format (metadata layout, attachments, reactions, multi-line messages, ...)
belongs in here.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Message:
    sender: str
    text: str
    timestamp: datetime | None = None   # optional; enables splitting into sessions


def parse_chat_history(path: Path) -> list[Message]:
    """Parse one chat export file (or directory) into messages, oldest first.

    Suggestions:
      * Normalise sender names (phone numbers / nicknames -> one display name).
      * Drop system messages ("X joined the group"), empty messages and media
        placeholders, or replace media with a short tag like "<image>".
      * Join multi-line messages into one `text` (newlines inside are fine).
      * Strip anything you don't want the model to learn (links, phone numbers).

    Example for lines like "2024-03-01 18:22 | Alice | hey what's up":

        messages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            ts, sender, text = line.split(" | ", 2)
            messages.append(Message(sender, text, datetime.strptime(ts, "%Y-%m-%d %H:%M")))
        return messages
    """
    raise NotImplementedError("Implement parse_chat_history() for your chat export format.")
