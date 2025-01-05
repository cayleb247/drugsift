import spacy
from spellchecker import SpellChecker
from multiprocessing import Pool
import csv
from itertools import chain
import config
import multiprocessing

def read_csv(file_path):
    '''
    Converts csv file to a list of table elements

    Parameters:
    file_path (str): Path to the csv file

    Returns:
    list: A list of table elements
    '''
    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]

def lemmatize_abstracts(abstracts):
    '''
    Use SpaCy's nlp object to tokenize, POS tag, and lemmatize abstract text

    Parameters:
    abstracts (list): List of all abstract text

    Returns:
    list: A list of lemma for each abstract
    '''
    multiprocessing.set_start_method('fork', force=True)

    nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"]) # SpaCy thingy

    lemmas = []

    for doc in nlp.pipe(abstracts, n_process=config.SPACY_NPROCESSES, batch_size=config.SPACY_BATCHES):
        lemmas.append([token.lemma_ for token in doc])
    
    return lemmas


def extract_clinical_features(abstract):
    '''
    Looks for clinical feature words from a single processed abstract

    Parameters:
    word (str): a single preprocessed word (lemma)

    Returns:
    list: a list of clinical features taken from the abstract
    '''
    suffixes = read_csv("Data/clinical_features.csv")

    clinical_features = []

    for word in abstract:
        for suffix in suffixes:
            if word.endswith(suffix):
                clinical_features.append(word)

    return clinical_features

def multiprocess(function, list_):
    '''
    Use multiprocessing pool to perform a function on a list

    Paramenters:
    function (func): function to be performed on inputted list
    list (list): list for function to be performed upon

    Returns:
    list: inputted list after function is mapped onto it
    '''

    if __name__ == "__main__":
        with Pool(config.POOL_PROCESSES) as p:
            output_list = p.map(function, list_)
        return list(chain.from_iterable(output_list))

