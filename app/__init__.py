# app/code.py
from flask import Flask 

def create_app():
    app = Flask(__name__)

    from .code import code
    app.register_blueprint(code, url_prefix='/')
    
    return app