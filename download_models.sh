#!/bin/bash

echo "📥 Downloading Models.zip from Google Drive..."
gdown --id 1abcDEFghiJKLmnopQRstuVWXYZ -O Models.zip

echo "🗜️ Unzipping Models.zip..."
unzip Models.zip -d Models

echo "🧹 Cleaning up zip file..."
rm Models.zip
