#!/bin/bash

# run this script once to set up the project

cd client
npm install

cd ../server
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

echo "Setup complete."
