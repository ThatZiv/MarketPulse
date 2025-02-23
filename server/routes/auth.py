# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# app/routes/auth_routes.py

import os
from flask import Blueprint, request, send_file
import flask_jwt_extended as jw
import requests
from routes.llm import llm_bp
auth_bp = Blueprint('auth', __name__)
LOGODEV_API_KEY = os.getenv('LOGODEV_API_KEY')

auth_bp.register_blueprint(llm_bp)

@auth_bp.route('/private', methods=['GET', 'POST'])
@jw.jwt_required()
def private():
    #current_user = jw.get_jwt_identity()
    return 'Private route'

@auth_bp.route('/logo', methods=['GET'])
@jw.jwt_required()
def ticker_logo():
    ticker = request.args.get('ticker')
    cache_dir = f"{os.getcwd()}/public/cache"
    if not ticker:
        return {"error": "Ticker parameter is required"}, 400
    # sanitize input
    ticker=ticker.replace('/', '').replace('..', '')\
        .replace(' ', '').replace('\\', '').replace('..', '')
    loc = f"{cache_dir}/{ticker}.png"
    if not os.path.exists(loc):
        base_url = "https://img.logo.dev/ticker/"
        url=f'{ticker}?token={LOGODEV_API_KEY}&size=300&format=png&fallback=monogram'
        r = requests.get(base_url+url, timeout=10)
        if r.status_code == 200:
            with open(loc, 'wb') as f:
                f.write(r.content)
        else:
            base_url = "https://ui-avatars.com/api/?name="
            url = f"{ticker}&format=png&size=300&background=a9a9a9&length=4"
            fallback = requests.get(base_url+url, timeout=5)
            with open(loc, 'wb') as f:
                f.write(fallback.content)
    return send_file(f'{cache_dir}/{ticker}.png', mimetype='image/png')
