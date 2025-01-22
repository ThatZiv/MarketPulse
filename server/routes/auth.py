# app/routes/auth_routes.py
from flask import Blueprint, request

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    return 'Logging in...'

@auth_bp.route('/logout')
def logout():
    return 'Logging out...'
