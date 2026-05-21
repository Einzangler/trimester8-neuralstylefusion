#!/bin/bash
echo "Neural Style Fusion - Web Interface"
echo "===================================="
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting server at http://localhost:7860"
python3 app.py
