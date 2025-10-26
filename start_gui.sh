#!/bin/bash

# Activate virtual environment and launch GUI
cd "$(dirname "$0")"
source venv/bin/activate
python gui_launcher.py
