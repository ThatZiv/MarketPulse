# app/routes/auth_routes.py
from flask import Blueprint, request
import flask_jwt_extended as jw

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/private', methods=['GET', 'POST'])
@jw.jwt_required()
def private():
    current_user = jw.get_jwt_identity()
    return 'Private route'

@auth_bp.route('/logout')
def logout():
    return 'Logging out...'
