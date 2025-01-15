from . import db
from sqlalchemy.types import JSON

class queryData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    abstract = db.Column(db.String(1000))
    date = db.Column(db.String(150))
    lemmas = db.Column (db.JSON)

    def __repr__(self):
        return f"<queryData(id={self.id}, search='{self.search}', abstract='{self.abstract}', date={self.date}, lemmas={self.lemmas})>"

class compoundScoringData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    compound_term = db.Column(db.String(100))
    aggregated_tfidf = db.Column(db.Float)

    def __repr__(self):
        return f"<compoundData(id={self.id}, search='{self.search}', term='{self.compound_term}', tfidf={self.aggregated_tfidf})>"
    
class featureScoringData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    feature_term = db.Column(db.String(100))
    aggregated_tfidf = db.Column(db.Float)

    def __repr__(self):
        return f"<featureData(id={self.id}, search='{self.search}', term='{self.feature_term}', tfidf={self.aggregated_tfidf})>"
