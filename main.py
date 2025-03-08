# from Misc.logging_ import logger
from flask import session
import os
import sys

from website.models import queryData
from website.models import featureScoringData
from website.models import compoundScoringData
from website.models import associatedDiseases
from website.models import cosineSimilarity

from website import db, create_app

import pandas as pd
from multiprocessing import Pool

from Data_Magagement.data_collection import get_pubmed_count, get_queried_abstracts
from Data_Magagement.data_processing import lemmatize_abstracts, ClinicalFeaturesExtractor, RxNormDrugCompoundExtractor, DiseaseTermsExtractor
from Data_Magagement.data_scoring import HybridTemporalTFIDF

from Models.word_2_vec import train_word2vec_model, top_related_drugs

from Drugs_Proteins.chembl_api import getDrugData

from Drugs_Proteins.uniprot_api import get_protein_accession, get_protein_sequence

from admet_ai import ADMETModel

# Add the SSM_DTA directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SSM_DTA'))

# Now import
from SSM_DTA.infer import SSM_DTA

current_dir = os.path.dirname(os.path.abspath(__file__))

def getLitData(search_query: str, n_process=8, email="calebtw10@gmail.com", user_remove_terms=[]):
    '''
    Processes an entire disease query

    Params:
    search_query (str): the user-inputted disease query
    n_process (int): the number of multiprocessing processes to use
    first_disease: whether the query is the first disease (original input) or the associated disease (secondary input), 1 for original, 0 for associated
    email (str): user's email for pubmed api
    user_remove_terms (list): a list of user-inputted terms to reduce noise in output
    
    '''
    remove_terms = {"diagnosis", "analysis"}

    if len(user_remove_terms) > 0: # check if user inputted any remove terms
        for term in user_remove_terms: # add terms to remove_terms set
            remove_terms.add(term)

    app = create_app()

    print("main() successfully called")

    query_count = get_pubmed_count(search_query, email)
    
    abstract_df = get_queried_abstracts(search_query)

    # place the number of retrived abstract in session
    retrieved_abstracts = query_count - abstract_df["abstract"].eq('').sum()
    total_abstracts = query_count

    abstract_df["retrieved_abstracts"] = retrieved_abstracts
    abstract_df["total_abstracts"] = total_abstracts

    print(f"Able to retrieve {retrieved_abstracts} abstracts")

    abstract_df["lemmas"] = lemmatize_abstracts(abstract_df["abstract"].to_list())

    # logger.info("Abstracts successfully retrived and lemmatized")

    records = abstract_df.to_dict(orient='records')

    # breakpoint()
    # place abstract data in SQLite
    with app.app_context():

        abstract_data = [queryData(search=record["search-query"], 
                                   abstract=record["abstract"], 
                                   year=record["year-published"], 
                                   lemmas=record["lemmas"], 
                                   retrieved=record["retrieved_abstracts"], 
                                   total=record["total_abstracts"]) for record in records]
        db.session.bulk_save_objects(abstract_data)
        db.session.commit()

    # logger.info("Abstracts successfully placed in db")

    # print(abstract_df["abstract"].tolist()[0])
    # breakpoint()

    # process feature data
    feature_extractor = ClinicalFeaturesExtractor(os.path.join(current_dir, 'Data', 'clinical_features.csv'), remove_terms)
    features = feature_extractor.extract_clinical_features(n_process, abstract_df["lemmas"].to_list())
    feature_intervals = abstract_df["year-published"].to_list()

    tfidf = HybridTemporalTFIDF(search_query, recency_weight=0.1)
    tfidf.fit(abstract_df['lemmas'].to_list(), feature_intervals)

    feature_df = tfidf.get_comprehensive_terms(list(features))

    # place scored feature data in SQLite
    feature_records = feature_df.to_dict(orient='records')

    # print(records[0]["total_score"])
    # breakpoint()

    with app.app_context():

        feature_data = [featureScoringData(search=record["search_query"], feature_term=record["term"], tfidf_score=record["total_score"]) for record in feature_records]
        db.session.bulk_save_objects(feature_data)
        db.session.commit()

    # logger.info("Features successfully placed in db")

    # process compound data
    compound_extractor = RxNormDrugCompoundExtractor("Data/rxnorm_drugs.csv", "Data/words.txt")
    compounds = compound_extractor.extract_drug_compounds(n_process, abstract_df["lemmas"].to_list())
    # compounds = check_if_drug(list(compounds)) # denoise compound list by checking for SMILES string

    compound_df = tfidf.get_comprehensive_terms(list(compounds))

    # place scored compound data in SQLite
    compound_records = compound_df.to_dict(orient='records')

    with app.app_context():

        compound_data = [compoundScoringData(search=record["search_query"], compound_term=record["term"], tfidf_score=record["total_score"]) for record in compound_records]
        db.session.bulk_save_objects(compound_data)
        db.session.commit()

    # logger.info("Compounds successfully placed in db")


    extractor = DiseaseTermsExtractor(
        search_query,
        min_ngram_size=2,
        max_ngram_size=5,
        min_frequency=2
    )

    # Set clinical features and process documents
    extractor.set_clinical_features(list(features))

    # print(abstract_df["lemmas"].head())
    # print(abstract_df["abstract"].head())

    extractor.fit(abstract_df["lemmas"])
    
    # Get disease terms
    disease_terms_df = extractor.get_disease_terms()

    records = disease_terms_df.to_dict(orient='records')

    with app.app_context():
        feature_data = [associatedDiseases(search=record["search_query"], disease_term=record["disease_term"], frequency=record["frequency"]) for record in records]
        db.session.bulk_save_objects(feature_data)
        db.session.commit()

def resetDatabase():

    # recreate database
    app = create_app()

    with app.app_context():

        db.drop_all()
        db.create_all() 

def runWord2Vec(search_term):

    app = create_app()

    # Configuration
    connection_string = f"sqlite:///{os.path.join(current_dir, '.', 'instance', 'data.db')}"
    abstract_column_name = "abstract"  # Replace with your actual column name
    
    # Train the model
    model = train_word2vec_model(connection_string, abstract_column_name)
    
    drug_df = top_related_drugs(model, search_term)

    records = drug_df.to_dict(orient='records')

    with app.app_context():
        feature_data = [cosineSimilarity(search=record["search_query"], term=record["term"], cosine_similarity=record["cosine_similarity"], corpus=record["corpus"]) for record in records]
        db.session.bulk_save_objects(feature_data)
        db.session.commit()


def protein_extactor(disease_id):
  '''
  Returns a list of amino acid sequences given a UniProt disease ID
  '''
  proteins = []

  protein_accessions = get_protein_accession(disease_id)

  for protein_accession in protein_accessions:
      protein_dict = {}
      protein_dict["chembl_accession"] = protein_accession
      protein_dict["sequence"] = get_protein_sequence(protein_accession)
      proteins.append(protein_dict)

  return proteins


def getAverageAffinity(scores: list):
    '''
    Find the average binding affinity of a drug to the disease's proteins
    '''
    scores = [float(score) for score in scores]

    return sum(scores) / len(scores)

def generateDrugProfiles(drug_list: list, disease_id: str):
    '''
    Generate the a drug profile including model predicted binding affinity, SMILES string, and ADMET
    '''
    predictor = SSM_DTA("Models/bindingdb_Ki_SSM.pt", "SSM_DTA/dict")

    drug_profiles = getDrugData(drug_list)

    proteins = protein_extactor(disease_id)

    # run prediction using SSM-DTA
    for drug in drug_profiles:
        if "SMILES" in drug:

            drug["scores"] = {}
            for protein in proteins:
                score = predictor.run_inference(drug["SMILES"], protein["sequence"])
                drug["scores"][protein["chembl_accession"]] = score
            drug["average_score"] = getAverageAffinity(list(drug["scores"].values()))
        
    
    return drug_profiles

# print(generateDrugProfiles(["benfotiamine", "eteplirsen", "ataluren", "trametinib", "clobetasol", "trimetazidine", "xaliproden", "taurursodiol", "levosimendan", "clemastine"], "DI-00001"))
# breakpoint()
            

# print(getDrugData(["benfotiamine", "eteplirsen", "ataluren", "trametinib", "clobetasol", "trimetazidine", "xaliproden", "taurursodiol", "levosimendan", "clemastine"]))
# breakpoint()
