#!/bin/bash

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
elif command -v py &>/dev/null; then
    PYTHON=py
elif command -v py3 &>/dev/null; then
    PYTHON=py3
else
    echo "Python is not installed."
    exit 1
fi

tmux new-session -A -s dev -d "cd server && source .venv/bin/activate && $PYTHON main.py"
tmux split-window -v 'cd client && npm run dev -- --host'
tmux attach-session -t dev
