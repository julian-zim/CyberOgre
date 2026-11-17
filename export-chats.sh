#!/bin/bash

source ~/.venv/temp/bin/activate
sigexport --source ~/snap/signal-desktop/current/.config/Signal --chats "Rote Oger" "$DATA"/CyberOgre/Chat/SignalExport #--no-attachments --no-stickers
rm -rf "$DATA/CyberOgre/Chat/SignalExport/exported_stickers" "$DATA/CyberOgre/Chat/SignalExport/RoteOger/media"
