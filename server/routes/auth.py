# app/routes/auth_routes.py
from flask import Blueprint, request, send_file
import flask_jwt_extended as jw
import requests
import os
auth_bp = Blueprint('auth', __name__)
LOGODEV_API_KEY = os.getenv('LOGODEV_API_KEY')

@auth_bp.route('/private', methods=['GET', 'POST'])
@jw.jwt_required()
def private():
    current_user = jw.get_jwt_identity()
    return 'Private route'

@auth_bp.route('/logo', methods=['GET'])
@jw.jwt_required()
def ticker_logo():
    ticker = request.args.get('ticker')
    cache_dir = f"{os.getcwd()}/public/cache"
    # sanitize input
    ticker = ticker.replace('/', '').replace('..', '')\
        .replace(' ', '').replace('\\', '').replace('..', '')
    loc = f"{cache_dir}/{ticker}.png"
    if not os.path.exists(loc):
        url = f'https://img.logo.dev/ticker/{ticker}?token={LOGODEV_API_KEY}&size=300&format=png&fallback=monogram'
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(loc, 'wb') as f:
                f.write(r.content)
        else:
            fallback = requests.get(\
                f"https://ui-avatars.com/api/?name={ticker}&format=png&size=300&background=a9a9a9&length=4", timeout=5)
            with open(loc, 'wb') as f:
                f.write(fallback.content)
    return send_file(f'{cache_dir}/{ticker}.png', mimetype='image/png')
