from website.models import queryData
from website.models import featureScoringData
from website.models import compoundScoringData
from website.models import associatedDiseases

from website import db, create_app

app = create_app()
with app.app_context():
    data = compoundScoringData.query.get(16)
    print(data)