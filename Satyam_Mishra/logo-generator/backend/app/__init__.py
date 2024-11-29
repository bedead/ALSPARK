from flask import Flask
from flask_cors import CORS
from .config import Config


def create_app():
    __name__ = 'AI Logo Generator'
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS
    CORS(app)

    from .routes.image_routes import image_bp
    app.register_blueprint(image_bp, url_prefix='/api/logo')

    return app
