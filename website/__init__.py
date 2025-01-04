from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///data.db"
    app.config['SECRET_KEY'] = "secret"
    db.init_app(app)

    from .views import views
    
    app.register_blueprint(views, url_prefix='/')

    from .models import Abstracts

    with app.app_context():
        db.create_all()

    return app
