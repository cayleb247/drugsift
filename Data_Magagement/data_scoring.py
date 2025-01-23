import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Union
from datetime import datetime
import pandas as pd

class TemporalTFIDF:
    def __init__(self, search_query: str, recency_weight: float = 0.1):
        """
        Initialize with temporal weighting for TF-IDF scores.
        
        Args:
            recency_weight: Weight factor for temporal scoring (higher means more emphasis on recency)
        """
        self.search_query = search_query
        self.corpus_docs = []
        self.interval_docs = defaultdict(list)
        self.term_scores = defaultdict(dict)
        self.recency_weight = recency_weight
        self.intervals = []
        
    def _normalize_date(self, date: Union[str, int]) -> int:
        """Convert date string to year."""
        if isinstance(date, str):
            try:
                return pd.to_datetime(date).year
            except:
                return int(date)
        return date
        
    def _calculate_temporal_weight(self, interval: int) -> float:
        """Calculate weight based on how recent the interval is."""
        max_year = max(self.intervals)
        years_from_present = max_year - interval
        return np.exp(-self.recency_weight * years_from_present)
    
    def fit(self, documents: List[List[str]], intervals: List[Union[str, int]]):
        """
        Fit the model on documents with their corresponding intervals.
        
        Args:
            documents: List of tokenized documents (each document is a list of strings)
            intervals: List of interval labels (e.g., years) corresponding to each document
        """
        # Normalize intervals to years
        normalized_intervals = [self._normalize_date(interval) for interval in intervals]
        self.intervals = sorted(set(normalized_intervals))
        
        # Store documents
        self.corpus_docs = documents
        for doc, interval in zip(documents, normalized_intervals):
            self.interval_docs[interval].append(doc)
            
    def _calculate_tf(self, term: str, document: List[str]) -> float:
        """Calculate term frequency for a given term in a document."""
        if not document:
            return 0.0
        term_count = Counter(document)[term]
        return term_count / len(document)
    
    def _calculate_df_corpus(self, term: str) -> float:
        """Calculate document frequency across the entire corpus."""
        if not self.corpus_docs:
            return 0.0
        docs_with_term = sum(1 for doc in self.corpus_docs if term in doc)
        return docs_with_term / len(self.corpus_docs)
    
    def transform_term(self, term: str, interval: Union[int, str]) -> float:
        """
        Calculate temporally weighted TF-IDF for a specific term in a given interval.
        """
        interval = self._normalize_date(interval)
        if interval not in self.interval_docs:
            return 0.0
            
        # Calculate interval-specific TF
        interval_docs = self.interval_docs[interval]
        tf_interval = np.mean([self._calculate_tf(term, doc) for doc in interval_docs])
        
        # Calculate corpus-wide DF
        df_corpus = self._calculate_df_corpus(term)
        if df_corpus == 0:
            return 0.0
            
        # Calculate document count ratios
        D_corpus = len(self.corpus_docs)
        D_interval = len(interval_docs)
        
        # Calculate temporal weight
        temporal_weight = self._calculate_temporal_weight(interval)
        
        # Calculate weighted TF-IDF
        score = tf_interval * (1 / df_corpus) * (D_corpus / D_interval) * temporal_weight
        return score
    
    def get_comprehensive_terms(self, terms: List[str]) -> pd.DataFrame:
        """
        Get comprehensive scoring for terms across all intervals.
        
        Args:
            terms: List of terms to analyze
            
        Returns:
            DataFrame with columns: term, total_score, max_score, latest_score, 
            peak_interval, intervals_present
        """
        results = []
        
        for term in terms:
            term_scores = {
                interval: self.transform_term(term, interval)
                for interval in self.intervals
            }
            
            # Calculate various metrics
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
                'scores_by_interval': term_scores
            })
        
        # Convert to DataFrame and sort by total_score
        df = pd.DataFrame(results)
        return df.sort_values('total_score', ascending=False)

# Example usage
# def example_usage():
#     # Sample data
#     documents = [
#         ["drug", "compound", "effect"],
#         ["novel", "drug", "treatment"],
#         ["clinical", "trial", "results"],
#         ["compound", "synthesis", "method"]
#     ]
#     intervals = [2020, 2020, 2021, 2021]
#     terms_of_interest = ["drug", "compound", "clinical"]
    
#     # Initialize and fit the model
#     tfidf = TemporalTFIDF(recency_weight=0.1)
#     tfidf.fit(documents, intervals)
    
#     # Get comprehensive term analysis
#     results = tfidf.get_comprehensive_terms(terms_of_interest)
    
#     return results

# print(example_usage())