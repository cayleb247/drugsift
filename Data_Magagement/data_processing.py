import spacy
from spellchecker import SpellChecker
from multiprocessing import Pool
import csv
from itertools import chain
import multiprocessing
import re
from string import punctuation
from spacy.lang.en import stop_words
from typing import List, Set, Dict
from collections import Counter
import nltk
from nltk.util import ngrams
import pandas as pd
from tqdm import tqdm


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
        lemmas.append([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    
    return lemmas

class ClinicalFeaturesExtractor:
    def __init__(self, features_file: str, remove_terms: set):
        '''
        Initialize extractor object with feature file path

        Parameters:
        features_file (str): Path to clinical features csv file
        remove_terms (set): A set of terms to remove to further denoise
        '''

        self.clinical_features_pattern = re.compile('(' + '|'.join(map(re.escape, self._read_csv(features_file))) + ')$', re.IGNORECASE)
        self.remove_terms = remove_terms
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
        return [word for word in abstract if self.clinical_features_pattern.search(word) and word not in self.remove_terms]
    
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
        return set(chain.from_iterable(results))

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
        return set(chain.from_iterable(results))
    
class DiseaseTermsExtractor:
    def __init__(self, 
                 search_query: str,
                 min_ngram_size: int = 2,
                 max_ngram_size: int = 5,
                 min_frequency: int = 5,
                 window_size: int = 5):
        """
        Initialize the disease terms extractor.
        
        Args:
            min_ngram_size: Minimum size of n-grams to consider
            max_ngram_size: Maximum size of n-grams to consider
            min_frequency: Minimum frequency threshold for n-grams
            window_size: Number of words to look at before and after clinical features
        """
        self.search_query = search_query
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size
        self.min_frequency = min_frequency
        self.window_size = window_size
        self.clinical_features = set()
        self.ngram_frequencies = Counter()
        
    def set_clinical_features(self, features: List[str]):
        """Set the clinical features to look for in the text."""
        self.clinical_features = set(features)
        
    def _extract_windows(self, tokens: List[str]) -> List[List[str]]:
        """Extract word windows around clinical features."""
        windows = []
        for i, token in enumerate(tokens):
            if token in self.clinical_features:
                # Get window of words before and after the clinical feature
                start = max(0, i - self.window_size)
                end = min(len(tokens), i + self.window_size + 1)
                window = tokens[start:end]
                windows.append(window)
        return windows
    
    def _generate_ngrams_from_window(self, window: List[str]) -> List[tuple]:
        """Generate n-grams from a window of words."""
        all_ngrams = []
        for n in range(self.min_ngram_size, self.max_ngram_size + 1):
            # Generate n-grams
            window_ngrams = list(ngrams(window, n))
            # Filter n-grams to ensure they contain at least one clinical feature
            valid_ngrams = [
                ng for ng in window_ngrams 
                if any(term in ng for term in self.clinical_features)
            ]
            all_ngrams.extend(valid_ngrams)
        return all_ngrams
    
    def fit(self, documents: List[str]):
        """
        Process documents and collect n-gram frequencies.
        
        Args:
            documents: List of document strings
        """
        self.ngram_frequencies = Counter()
        
        for doc in tqdm(documents, desc="Processing documents"):
            doc = ' '.join(doc)
            # Tokenize document
            tokens = nltk.word_tokenize(doc.lower())
            
            # Extract windows around clinical features
            windows = self._extract_windows(tokens)
            
            # Generate and count n-grams from each window
            for window in windows:
                ngrams_list = self._generate_ngrams_from_window(window)
                self.ngram_frequencies.update(ngrams_list)
    
    def get_disease_terms(self) -> pd.DataFrame:
        """
        Get the filtered disease terms and their frequencies.
        
        Returns:
            DataFrame with columns: disease_term, frequency
        """
        # Filter n-grams by frequency threshold
        frequent_ngrams = {
            ngram: freq 
            for ngram, freq in self.ngram_frequencies.items() 
            if freq >= self.min_frequency
        }
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame([
            {
                'search_query': self.search_query,
                'disease_term': ' '.join(ngram),
                'frequency': freq
            }
            for ngram, freq in frequent_ngrams.items()
        ])
        
        if not df.empty:
            df = df.sort_values('frequency', ascending=False)
        
        return df