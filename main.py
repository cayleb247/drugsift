# from Misc.logging_ import logger

from website.models import queryData
from website.models import featureScoringData
from website.models import compoundScoringData

from website import db, create_app

import pandas as pd
from multiprocessing import Pool

from Data_Magagement.data_collection import get_pubmed_count, get_queried_abstracts
from Data_Magagement.data_processing import lemmatize_abstracts, ClinicalFeaturesExtractor, DrugCompoundExtractor, DiseaseTermsExtractor
from Data_Magagement.data_scoring import HybridTemporalTFIDF

remove_terms = {"diagnosis", "analysis"}

def getLitData(search_query: str, n_process:int, email=None):
    app = create_app()

    # logger.info("main() successfully called")

    query_count = get_pubmed_count(search_query, email)
    
    abstract_df = get_queried_abstracts(search_query)
    retrieved_count = query_count - abstract_df["abstract"].eq('').sum()

    # logger.info(f"Able to retrieve {retrieved_count} abstracts")

    abstract_df["lemmas"] = lemmatize_abstracts(abstract_df["abstract"].to_list())

    # print(abstract_df.head())
    # breakpoint()

    # logger.info("Abstracts successfully retrived and lemmatized")

    records = abstract_df.to_dict(orient='records')

    # print(records[0])
    # breakpoint()
    # place abstract data in SQLite
    with app.app_context():

        abstract_data = [queryData(search=record["search-query"], abstract=record["abstract"], year=record["year-published"], lemmas=record["lemmas"]) for record in records]
        db.session.bulk_save_objects(abstract_data)
        db.session.commit()

    # logger.info("Abstracts successfully placed in db")

    # print(abstract_df["abstract"].tolist()[0])
    # breakpoint()

    # process feature data
    feature_extractor = ClinicalFeaturesExtractor("Data/clinical_features.csv", remove_terms)
    features = feature_extractor.extract_clinical_features(n_process, abstract_df["lemmas"].to_list())
    feature_intervals = abstract_df["year-published"].to_list()

    tfidf = HybridTemporalTFIDF(search_query, recency_weight=0.1)
    tfidf.fit(abstract_df['lemmas'].to_list(), feature_intervals)

    feature_df = tfidf.get_comprehensive_terms(list(features))

    print(feature_df.head())
    breakpoint()

    # place scored feature data in SQLite
    records = feature_df.to_dict(orient='records')

    # print(records[0]["total_score"])
    # breakpoint()

    with app.app_context():

        feature_data = [featureScoringData(search=record["search_query"], feature_term=record["term"], tfidf_score=record["total_score"]) for record in records]
        db.session.bulk_save_objects(feature_data)
        db.session.commit()

    # logger.info("Features successfully placed in db")

    # process compound data
    compound_extractor = DrugCompoundExtractor("Data/stems.csv", "Data/words.txt")
    compounds = compound_extractor.extract_drug_compounds(n_process, abstract_df["lemmas"].to_list())

    print(list(compounds)[:10])
    breakpoint()

    # tfidf.fit(abstract_df['lemmas'].to_list(), feature_intervals)

    compound_df = tfidf.get_comprehensive_terms(list(compounds))

    print(compound_df.head(100))
    bob_row = compound_df.loc[compound_df['term'] == 'riociguat']
    print(bob_row)
    breakpoint()

    # place scored compound data in SQLite
    records = compound_df.to_dict(orient='records')

    with app.app_context():

        compound_data = [compoundScoringData(search=record["search_query"], compound_term=record["term"], tfidf_score=record["total_score"]) for record in records]
        db.session.bulk_save_objects(compound_data)
        db.session.commit()

    # logger.info("Compounds successfully placed in db")

    extractor = DiseaseTermsExtractor(
        min_ngram_size=2,
        max_ngram_size=5,
        min_frequency=2
    )

     # Set clinical features and process documents
    extractor.set_clinical_features(list(features))

    print(abstract_df["lemmas"].head())
    print(abstract_df["abstract"].head())
    breakpoint()
    extractor.fit(abstract_df["lemmas"])
    
    # Get disease terms
    disease_terms_df = extractor.get_disease_terms()

    print(disease_terms_df.head(10))


getLitData("chronic thromboembolic pulmonary hypertension AND english[Language]", 8, "calebtw8@gmail.com")