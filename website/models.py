from . import db
from sqlalchemy.types import JSON

class queryData(db.Model):
    __tablename__ = "queryData"

    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    abstract = db.Column(db.String(1000))
    year = db.Column(db.String(150))
    lemmas = db.Column (db.JSON)
    retrieved = db.Column(db.String(100))
    total = db.Column(db.String(100))

    def __repr__(self):
        return f"<queryData(id={self.id}, search='{self.search}', abstract='{self.abstract}', year={self.year}, lemmas={self.lemmas}, retrieved_abstracts={self.retrieved}, total_abstracts={self.total})>"

class compoundScoringData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    compound_term = db.Column(db.String(100))
    tfidf_score = db.Column(db.Float)

    def __repr__(self):
        return f"<compoundScoringData(id={self.id}, search='{self.search}', term='{self.compound_term}', tfidf_score={self.tfidf_score})>"
    
class featureScoringData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    feature_term = db.Column(db.String(100))
    tfidf_score = db.Column(db.Float)

    def __repr__(self):
        return f"<featureScoringData(id={self.id}, search='{self.search}', term='{self.feature_term}', tfidf_score={self.tfidf_score})>"
    
class associatedDiseases(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    disease_term = db.Column(db.String(100))
    frequency = db.Column(db.Float)

    def __repr__(self):
        return f"<associatedDiseases(id={self.id}, search='{self.search}', disease_term='{self.disease_term}', frequency={self.frequency})>"
    
class cosineSimilarity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    term = db.Column(db.String(100))
    cosine_similarity = db.Column(db.Float)

    def __repr__(self):
        return f"<cosineSimilarity(id={self.id}, search='{self.search}', term='{self.term}', cosine_similarity={self.cosine_similarity})>"