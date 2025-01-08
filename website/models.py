from . import db

class Abstracts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    abstract = db.Column(db.String(1000))


