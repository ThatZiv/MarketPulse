from flask import Flask, session, redirect, url_for, request
from dotenv import load_dotenv
import os
load_dotenv()

def create_app():
    app = Flask(__name__)

    app.secret_key = os.getenv('TEST_KEY')
    

    from routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    return app


if __name__ == '__main__':  
    app = create_app()
    # @app.route("/")
    # def hello_world():
    #     return "index page"
    app.run(debug=True, host='0.0.0.0')