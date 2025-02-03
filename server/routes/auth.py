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
def ticker_logo(): # TODO: fix the 422 errors - why does jwt not work?
    ticker = request.args.get('ticker')
    print(ticker)
    cache_dir = "public/cache"
    # sanitize input
    ticker = ticker.replace('/', '').replace('..', '').replace(' ', '').replace('\\', '').replace('..', '')
    loc = f"{cache_dir}/{ticker}.png"
    if not os.path.exists(loc):
        url = f'https://img.logo.dev/ticker/NKE?token=${LOGODEV_API_KEY}&size=300&format=png&retina=true'
        r = requests.get(url)
        if r.status_code == 200:
            with open(loc, 'wb') as f:
                f.write(r.content)
        else:
            fallback = requests.get(f"https://ui-avatars.com/api/?name={ticker}")
            with open(loc, 'wb') as f:
                f.write(fallback.content)
    return send_file(f'{cache_dir}/{ticker}.png', mimetype='image/png')


@auth_bp.route('/logout')
def logout():
    return 'Logging out...'
