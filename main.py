# from Misc.logging_ import logger

from website.models import queryData
from website.models import featureScoringData
from website.models import compoundScoringData

from website import db, create_app

import pandas as pd
from multiprocessing import Pool


from Data_Magagement.data_collection import get_pubmed_count, get_queried_abstracts
from Data_Magagement.data_processing import lemmatize_abstracts, ClinicalFeaturesExtractor, DrugCompoundExtractor
from Data_Magagement.data_scoring import tfidfScorer

def getLitData(query: str, n_process:int, email=None):
    app = create_app()

    # logger.info("main() successfully called")

    query_count = get_pubmed_count(query, email)
    
    abstract_df = get_queried_abstracts(query)
    retrieved_count = query_count - abstract_df["abstract"].eq('').sum()

    # logger.info(f"Able to retrieve {retrieved_count} abstracts")

    abstract_df["lemmas"] = lemmatize_abstracts(abstract_df["abstract"].to_list())

    # logger.info("Abstracts successfully retrived and lemmatized")

    records = abstract_df.to_dict(orient='records')

    print(records[0])
    breakpoint()
    # place abstract data in SQLite
    with app.app_context():

        abstract_data = [queryData(search=record["search-query"], abstract=record["abstract"], date=record["date-published"], lemmas=record["lemmas"]) for record in records]
        db.session.bulk_save_objects(abstract_data)
        db.session.commit()

    # logger.info("Abstracts successfully placed in db")

    breakpoint()

    # process feature data
    feature_extractor = ClinicalFeaturesExtractor("Data/clinical_features.csv")
    features = feature_extractor.extract_clinical_features(n_process, abstract_df["abstract"].to_list())

    feature_scorer = tfidfScorer(abstract_df["lemmas"], features)
    feature_df = feature_scorer.aggregate_score(query)

    print(feature_df.head())
    breakpoint()

    # place scored feature data in SQLite
    records = feature_df.to_dict(orient='records')

    with app.app_context():

        feature_data = [featureScoringData(search=record["search-query"], feature_term=record["term"], aggregated_tfidf=record["aggregate_score"]) for record in records]
        db.session.bulk_save_objects(feature_data)
        db.session.commit()

    # logger.info("Features successfully placed in db")

    # process compound data
    compound_extractor = DrugCompoundExtractor("Data/stems.csv", "Data/words.txt")
    compounds = compound_extractor.extract_drug_compounds(n_process, abstract_df["abstract"].to_list())

    compound_scorer = tfidfScorer(abstract_df["lemmas"], compounds)
    compound_df = compound_scorer.aggregate_score(query)

    # place scored compound data in SQLite
    records = compound_df.to_dict(orient='records')

    with app.app_context():

        compound_data = [compoundScoringData(search=record["search-query"], feature_term=record["term"], aggregated_tfidf=record["aggregate_score"]) for record in records]
        db.session.bulk_save_objects(compound_data)
        db.session.commit()

    # logger.info("Compounds successfully placed in db")


    