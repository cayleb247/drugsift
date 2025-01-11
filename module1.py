from website.models import queryData
from website import db

import pandas as pd
from multiprocessing import Pool


from Data_Magagement.data_collection import get_pubmed_count, get_queried_abstracts
from Data_Magagement.data_processing import lemmatize_abstracts, ClinicalFeaturesExtractor, DrugCompoundExtractor
from Data_Magagement.data_scoring import tfidfScorer

def main(query: str, n_process:int, email=None):

    query_count = get_pubmed_count(query, email)
    
    df = get_queried_abstracts(query)
    retrieved_count = query_count - df["abstract"].eq('').sum()

    df["lemmas"] = lemmatize_abstracts(df["abstract"].to_list())

    records = df.to_dict(orient='records')

    # place data in SQLite
    data = [queryData(search=record["search-query"], abstract=record["abstract"], date=record["data-published"], lemmas=record["lemmas"]) for record in records]
    db.session.bulk_save_objects(data)
    db.session.commit()

def features_to_db(query:str, n_process:int):
    extractor = ClinicalFeaturesExtractor("")

if __name__ == "__main__":
      
    main("chronic thromboembolic pulmonary hypertension AND english[Language]", 5, "calebtw8@gmail.com")

    print(df.head())
    print(retrieved_count)


    




    