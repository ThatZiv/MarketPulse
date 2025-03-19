#Created as a test server to allow for testing api's without adding them to the main server
from dotenv import load_dotenv
import threading
from flask import Flask, request, jsonify, Response
import os
from load_data import load_stocks
from sqlalchemy import create_engine, select, func

from models.run_models import run_models

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


