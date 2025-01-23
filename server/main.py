from flask import Flask, session, redirect, url_for, request
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import flask_jwt_extended as jw
from flask_cors import CORS


load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def create_app():
    app = Flask(__name__)
    CORS(app)
    jwt = jw.JWTManager()
    jwt.init_app(app)   

    app.secret_key = os.getenv('TEST_KEY')
    

    from routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    return app


if __name__ == '__main__':  
    app = create_app()
    @app.route('/api/private')
    @jw.jwt_required()
    def private_route():
        current_user = jw.get_jwt_identity()
  

        return 'message: This is a private route'

    app.run(debug=True, host='0.0.0.0')