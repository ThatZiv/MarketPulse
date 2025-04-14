# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import threading
import os
from flask import Flask
from load_data import load_stocks
from models.run_models import run_models

# Manualy add daya and run the models.
# This is setup as a server so it will run the
# same way as it does on the main server.
# Flask app closes after running the models.
# Still needs to run same day as data added.
def create_app():
    app_1 = Flask(__name__)
    app_1.config["JWT_SECRET_KEY"] = os.environ.get("SUPABASE_JWT")
    app_1.config["MUTEX"] = threading.Lock()
    return app_1


if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        load_stocks()
        run_models()
