#!/bin/bash

tmux new-session -A -s dev -d 'cd server && source .venv/bin/activate && python main.py'
tmux split-window -v 'cd client && npm run dev -- --host'
tmux attach-session -t dev
