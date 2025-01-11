from . import db

class queryData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    abstract = db.Column(db.String(1000))
    date = db.Column(db.String(150))
    lemmas = db.Column (db.String(1000))

class drugScoringData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    drug_compound = db.Column(db.String(100))
    aggregated_tfidf = db.Column(db.Float)
    
class featureScoringData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search = db.Column(db.String(150))
    clinical_feature = db.Column(db.String(100))
    aggregated_tfidf = db.Column(db.Float)
