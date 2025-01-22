from flask import Flask

def create_app():
    app = Flask(__name__)

    from routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    return app


if __name__ == '__main__':  
    app = create_app()
    # @app.route("/")
    # def hello_world():
    #     return "index page"
    app.run(debug=True, host='0.0.0.0')