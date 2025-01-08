import spacy
from spellchecker import SpellChecker
from multiprocessing import Pool
import csv
from itertools import chain
import multiprocessing
import re

def lemmatize_abstracts(abstracts, n_process=2, batch_size=1000):
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

    for doc in nlp.pipe(abstracts, n_process=n_process, batch_size=batch_size):
        lemmas.append([token.lemma_ for token in doc])
    
    return lemmas

class ClinicalFeaturesExtractor:
    def __init__(self, features_file: str):
        '''
        Initialize extractor object with feature file path

        Parameters:
        features_file (str): Path to clinical features csv file
        '''

        self.clinical_features_pattern = re.compile('(' + '|'.join(map(re.escape, self._read_csv(features_file))) + ')$', re.IGNORECASE)

    def _read_csv(self, file_path: str):
        '''
        Converts csv file to a list of table elements

        Parameters:
        file_path (str): Path to the csv file

        Returns:
        set: A set of table elements
        '''
        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            return set([row[0] for row in reader])
        
    def _process_abstract(self, abstract: list):
        '''
        Process single abstract, checking against stems and dictionary to find drug compound terms.

        Parameters:
        abstract (list): list of processed words in abstract

        Returns:
        list: list of drug terms
        '''
        return [word for word in abstract if self.clinical_features_pattern.search(word)]
    
    def extract_clinical_features(self, n_process: int, abstracts: list):
        '''
        Use multiprocessing and _process_abstract method to extract drug compounds

        Parameters:
        n_process: the number of processes to be used when multiprocessing
        abstracts: a list of preprocessed abstracts

        Returns:
        list: a list of drug compounds taken from abstracts
        '''

        with Pool(n_process) as p:
            results = p.map(self._process_abstract, abstracts)
        return list(chain.from_iterable(results))

class DrugCompoundExtractor:
    def __init__(self, stems_file: str, words_file: str):
        '''
        Initialize extractor object with stem and word files paths

        Parameters:
        stems_file (str): Path to drug stems CSV file
        words_file (str): Path to text file containing dictionary
        '''

        # use regular expression pattern for efficient lookup
        self.drug_stems_pattern = re.compile('|'.join(map(re.escape, self._read_csv(stems_file))))

        # use a set to prevent checking words twice
        self.spell = SpellChecker()
        self.spell.word_frequency.load_text_file(words_file)
        self.known_words = set(self.spell.word_frequency.keys())


    def _read_csv(self, file_path: str):
        '''
        Converts csv file to a list of table elements

        Parameters:
        file_path (str): Path to the csv file

        Returns:
        set: A set of table elements
        '''
        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            return set([row[0].lower() for row in reader])
    
    def _process_abstract(self, abstract: list):
        '''
        Process single abstract, checking against stems and dictionary to find drug compound terms.

        Parameters:
        abstract (list): list of processed words in abstract

        Returns:
        list: list of drug terms
        '''
        return [word for word in abstract if self.drug_stems_pattern.search(word.lower()) 
            and word.lower() not in self.known_words]
    
    def extract_drug_compounds(self, n_process: int, abstracts: list):
        '''
        Use multiprocessing and _process_abstract method to extract drug compounds

        Parameters:
        n_process: the number of processes to be used when multiprocessing
        abstracts: a list of preprocessed abstracts

        Returns:
        list: a list of drug compounds taken from abstracts
        '''
        # if n_process is defined, if not, number of cpus minus 1 is da

        with Pool(n_process) as p:
            results = p.map(self._process_abstract, abstracts)
        return list(chain.from_iterable(results))
    def __init__(self, stems_file: str, words_file: str):
        '''
        Initialize extractor object with stem and word files paths

        Parameters:
        stems_file (str): Path to drug stems CSV file
        words_file (str): Path to text file containing dictionary
        '''

        # use regular expression pattern for efficient lookup
        self.drug_stems_pattern = re.compile('|'.join(map(re.escape, self._read_csv(stems_file))))

        # use a set to prevent checking words twice
        self.spell = SpellChecker()
        self.spell.word_frequency.load_text_file(words_file)
        self.known_words = set(self.spell.word_frequency.keys())


    def _read_csv(self, file_path: str):
        '''
        Converts csv file to a list of table elements

        Parameters:
        file_path (str): Path to the csv file

        Returns:
        set: A set of table elements
        '''
        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            return set([row[0].lower() for row in reader])
    
    def _process_abstract(self, abstract: list):
        '''
        Process single abstract, checking against stems and dictionary to find drug compound terms.

        Parameters:
        abstract (list): list of processed words in abstract

        Returns:
        list: list of drug terms
        '''
        return [word for word in abstract if self.drug_stems_pattern.search(word.lower()) 
            and word.lower() not in self.known_words]

    def extract_drug_compounds(self, n_process: int, abstracts: list):
        '''
        Use multiprocessing and _process_abstract method to extract drug compounds

        Parameters:
        n_process: the number of processes to be used when multiprocessing
        abstracts: a list of preprocessed abstracts

        Returns:
        list: a list of drug compounds taken from abstracts
        '''

        with Pool(n_process) as p:
            results = p.map(self._process_abstract, abstracts)
        return list(chain.from_iterable(results))
    
