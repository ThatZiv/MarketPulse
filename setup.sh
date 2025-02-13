#!/bin/bash

# run this script once to set up the project

cd client
npm install

cd ../server

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

$PYTHON -m venv .venv
source .venv/bin/activate

$PYTHON -m pip install -r requirements.txt

echo "Setup complete."
