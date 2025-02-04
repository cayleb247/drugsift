from gensim.parsing.preprocessing import remove_stopwords
from gensim import utils
import gensim.models
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from website.models import queryData
from website.models import cosineSimilarity
from website.models import compoundScoringData

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import your existing database model
# from ..website.models import queryData  # Replace with your actual model import

class MyCorpus:
    """An iterator that yields sentences (lists of str) from a SQL database."""
    def __init__(self, connection_string, abstract_column_name):
        """Initialize with database connection string and column name."""
        self.engine = create_engine(connection_string)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.abstract_column_name = abstract_column_name
    
    def __iter__(self):
        """Iterate over abstracts in the database."""
        # Query all abstracts using the specified column
        for record in self.session.query(queryData).yield_per(100):  # Process in batches of 100
            abstract_text = getattr(record, self.abstract_column_name)
            if abstract_text:
                # Remove stopwords and tokenize the text
                cleaned_text = remove_stopwords(abstract_text)
                yield utils.simple_preprocess(cleaned_text)
    
    def close(self):
        """Close the database session."""
        self.session.close()

def train_word2vec_model(connection_string, abstract_column_name):
    """Train a Word2Vec model using abstracts from the database."""
    # Create an instance of MyCorpus
    sentences = MyCorpus(connection_string, abstract_column_name)
    
    try:
        # Train the Word2Vec model using the sentences
        model = gensim.models.Word2Vec(sentences=sentences)
        return model
    finally:
        # Ensure database connection is closed
        sentences.close()

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    try:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except:
        return None

def analyze_word_similarity(model, word1, word2):
    """Analyze similarity between two words in the model."""
    if word1 in model.wv and word2 in model.wv:
        vec1 = model.wv[word1]
        vec2 = model.wv[word2]
        similarity = cosine_similarity(vec1, vec2)
        print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")
    else:
        if word1 not in model.wv:
            print(f"'{word1}' is not in the vocabulary.")
        if word2 not in model.wv:
            print(f"'{word2}' is not in the vocabulary.")

def print_top_words(model, n=10):
    """Print the top N most frequent words in the vocabulary."""
    print(f"Top {n} words in the vocabulary (stopwords removed):")
    for index, word in enumerate(model.wv.index_to_key):
        if index == n:
            break
        print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

def top_related_drugs(model, search_term: str):

    df = pd.DataFrame([
            {
                'search_query': search_term,
                'term': word,
                'cosine_similarity': cosine_similarity(model.wv[search_term], model.wv[word])
            }
            for word in [item.compound_term for item in compoundScoringData.query.all() if item.compound_term in set(model.wv.key_to_index.keys())]
        ])
    return df

