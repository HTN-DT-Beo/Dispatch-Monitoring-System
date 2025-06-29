#!/usr/bin/env bash
set -e

FOLDER_ID="1wlRoW0Cp-XeQK550w4agtjz3OFXX-UiS"   # ID folder thật

echo "📥 Downloading Google‑Drive folder ..."
gdown --folder "$FOLDER_ID" -O Models          # sẽ tạo ./Models/...

echo "✅ Done"
