import numpy as np
from collections import defaultdict
from typing import List, Dict, Union
from datetime import datetime
import pandas as pd
from gensim import corpora
from gensim.models import TfidfModel
from nltk.util import ngrams

class HybridTemporalTFIDF:
    def __init__(self, search_query: str, recency_weight: float = 0.1, n_gram: int = 1, min_freq: int = 2):
        """
        Initialize hybrid TFIDF model combining Gensim's implementation with temporal analysis.
        
        Args:
            search_query: The search query associated with this analysis
            recency_weight: Weight factor for temporal scoring
            n_gram: Size of n-grams to use
            min_freq: Minimum frequency threshold for terms
        """
        self.search_query = search_query
        self.recency_weight = recency_weight
        self.n_gram = n_gram
        self.min_freq = min_freq
        
        # Storage for documents and models
        self.corpus_docs = []
        self.interval_docs = defaultdict(list)
        self.intervals = []
        self.dictionary = None
        self.tfidf_model = None
        self.idf_dict = {}
        self.year_dict = {}
        
    def _normalize_date(self, date: Union[str, int]) -> int:
        """Convert date string to year."""
        if isinstance(date, str):
            try:
                return pd.to_datetime(date).year
            except:
                return int(date)
        return date
    
    def _process_ngrams(self, documents: List[List[str]]) -> List[List[str]]:
        """Process documents into n-grams if n_gram > 1."""
        if self.n_gram > 1:
            return [[" ".join(ngram) for ngram in ngrams(doc, self.n_gram)] for doc in documents]
        return documents
    
    def _calculate_temporal_weight(self, interval: int) -> float:
        """Calculate weight based on how recent the interval is."""
        max_year = max(self.intervals)
        years_from_present = max_year - interval
        return np.exp(-self.recency_weight * years_from_present)
    
    def fit(self, documents: List[List[str]], intervals: List[Union[str, int]]):
        """
        Fit the model on documents with their corresponding intervals.
        
        Args:
            documents: List of tokenized documents
            intervals: List of interval labels corresponding to each document
        """
        # Normalize intervals and store them
        normalized_intervals = [self._normalize_date(interval) for interval in intervals]
        self.intervals = sorted(set(normalized_intervals))
        
        # Process n-grams
        processed_docs = self._process_ngrams(documents)
        
        # Create and filter dictionary
        self.dictionary = corpora.Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=self.min_freq)
        
        # Convert to BOW format
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        # Create TFIDF model
        self.tfidf_model = TfidfModel(corpus=bow_corpus, dictionary=self.dictionary, smartirs='ntc')
        
        # Store documents by interval
        self.corpus_docs = processed_docs
        for doc, interval in zip(processed_docs, normalized_intervals):
            self.interval_docs[interval].append(doc)
        
        # Extract IDF values
        self.idf_dict = {self.dictionary[id]: val for id, val in self.tfidf_model.idfs.items()}
        
        # Calculate year-wise frequencies
        self.year_dict = {
            'idf': self.idf_dict,
            **{year: {} for year in self.intervals}
        }
        
        # Calculate frequencies for each year
        for doc, year, bow_doc in zip(processed_docs, normalized_intervals, bow_corpus):
            year_freqs = self.year_dict[year]
            doc_tfidf = dict(self.tfidf_model[bow_doc])
            
            for term_id, tfidf_value in doc_tfidf.items():
                term = self.dictionary[term_id]
                idf = self.idf_dict[term]
                tf = tfidf_value / idf if idf != 0 else 0
                year_freqs[term] = year_freqs.get(term, 0) + tf
    
    def transform_term(self, term: str, interval: Union[int, str]) -> float:
        """
        Calculate temporally weighted TFIDF for a term in a given interval.
        """
        interval = self._normalize_date(interval)
        if interval not in self.year_dict or term not in self.idf_dict:
            return 0.0
        
        # Get raw frequency and IDF
        tf = self.year_dict[interval].get(term, 0)
        idf = self.idf_dict[term]
        
        # Apply temporal weighting
        temporal_weight = self._calculate_temporal_weight(interval)
        
        # Calculate final score
        return tf * idf * temporal_weight
    
    def get_comprehensive_terms(self, terms: List[str]) -> pd.DataFrame:
        """
        Get comprehensive scoring for terms across all intervals.
        """
        results = []
        
        for term in terms:
            if term not in self.idf_dict:
                continue
                
            term_scores = {
                interval: self.transform_term(term, interval)
                for interval in self.intervals
            }
            
            # Calculate metrics
            total_score = sum(term_scores.values())
            max_score = max(term_scores.values())
            latest_score = term_scores[max(self.intervals)]
            peak_interval = max(term_scores.items(), key=lambda x: x[1])[0]
            intervals_present = sum(1 for score in term_scores.values() if score > 0)
            
            results.append({
                'search_query': self.search_query,
                'term': term,
                'total_score': total_score,
                'max_score': max_score,
                'latest_score': latest_score,
                'peak_interval': peak_interval,
                'intervals_present': intervals_present,
                'scores_by_interval': term_scores,
                'idf': self.idf_dict[term]
            })
        
        return pd.DataFrame(results).sort_values('total_score', ascending=False)
    

# import numpy as np
# from collections import Counter, defaultdict
# from typing import List, Dict, Union
# import pandas as pd

# class TemporalTFIDF:
#     def __init__(self, search_query, recency_weight: float = 0.1):
#         self.corpus_docs = []
#         self.interval_docs = defaultdict(list)
#         self.recency_weight = recency_weight
#         self.intervals = []
#         self.search_query = search_query
        
#     def fit(self, documents: List[List[str]], intervals: List[Union[str, int]]):
#         self.intervals = sorted(set(intervals))
#         self.corpus_docs = documents
#         for doc, interval in zip(documents, intervals):
#             self.interval_docs[interval].append(doc)
    
#     def _calculate_tf(self, term: str, document: List[str]) -> float:
#         if not document:
#             return 0.0
#         term = term.lower()
#         document = [word.lower() for word in document]
        
#         term_count = document.count(term)
#         return term_count / len(document)
    
#     def _calculate_df_corpus(self, term: str) -> float:
#         if not self.corpus_docs:
#             return 0.0
#         docs_with_term = sum(1 for doc in self.corpus_docs if term in doc)
#         return docs_with_term / len(self.corpus_docs)
    
#     def _calculate_temporal_weight(self, interval: int) -> float:
#         max_year = max(self.intervals)
#         years_from_present = int(max_year) - int(interval)
#         return np.exp(-self.recency_weight * years_from_present)
    
#     def get_term_scores(self, terms: List[str]) -> pd.DataFrame:
#         """Get comprehensive scoring for terms with sorted results."""
#         results = []
        
#         for term in terms:
#             total_score = 0
#             interval_scores = {}
            
#             for interval in self.intervals:
#                 interval_docs = self.interval_docs[interval]
#                 tf = np.mean([self._calculate_tf(term, doc) for doc in interval_docs])
#                 df = self._calculate_df_corpus(term)
#                 temporal_weight = self._calculate_temporal_weight(interval)
                
#                 # Calculate interval score
#                 if df > 0:
#                     interval_score = tf * (1/df) * temporal_weight
#                     total_score += interval_score
#                     interval_scores[f'score_{interval}'] = interval_score
#                 else:
#                     interval_scores[f'score_{interval}'] = 0.0
            
#             # Add metadata
#             result = {
#                 'search': self.search_query,
#                 'term': term,
#                 'total_score': total_score,
#                 'doc_frequency': self._calculate_df_corpus(term),
#                 'doc_count': sum(1 for doc in self.corpus_docs if term in doc),
#                 **interval_scores
#             }
#             results.append(result)
        
#         # Convert to DataFrame and sort by total_score in descending order
#         df = pd.DataFrame(results)
#         return df.sort_values('total_score', ascending=False)

# # Example usage
# def example():
#     documents = [
#         ["drug", "compound", "effect"],
#         ["novel", "drug", "treatment"],
#         ["clinical", "trial", "results"],
#         ["compound", "synthesis", "method"]
#     ]
#     intervals = [2020, 2020, 2021, 2021]
#     terms = ["clinical", "compound", "drug", "effect", "treatment"]
    
#     tfidf = TemporalTFIDF(recency_weight=0.1)
#     tfidf.fit(documents, intervals)
    
#     # Get sorted results
#     results = tfidf.get_term_scores(terms)
#     return results