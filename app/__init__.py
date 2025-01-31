from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import and register blueprints
    from .api_chatbot import api_chatbot_bp
    app.register_blueprint(api_chatbot_bp, url_prefix='/api')

    return app