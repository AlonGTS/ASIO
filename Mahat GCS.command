#!/bin/bash
# Mahat GCS — double-click to launch the Ground Control Station
# Pi IP is read automatically from config.toml

cd "$(dirname "$0")"   # always run from the ASIO folder, regardless of where it's launched from
/opt/homebrew/bin/python3 gcs.py
