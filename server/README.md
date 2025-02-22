# MarketPulse web service

This is a web service that provides a RESTful API for the web UI using Flask.

## Installation

1. Install Python

2. Setup venv

```bash
cd server

python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt

# you may have to install these
sudo apt-get install libpq-dev postgresql-client
```

4. Run the server

```bash
python main.py
```
