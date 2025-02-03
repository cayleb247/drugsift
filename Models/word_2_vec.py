# from flask import Flask, jsonify
# from gensim import utils
# from gensim.parsing.preprocessing import remove_stopwords
# import gensim.models
# import numpy as np
# import sqlite3
# from typing import List
# import pandas as pd

# app = Flask(__name__)

# class SQLiteCorpus:
#     """An iterator that yields sentences (lists of str) from SQLite database."""
#     def __init__(self, db_path: str, table_name: str, text_column: str):
#         self.db_path = db_path
#         self.table_name = table_name
#         self.text_column = text_column

#     def __iter__(self):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             # Query all text data from the specified table and column
#             cursor.execute(f"SELECT {self.text_column} FROM {self.table_name}")
            
#             for (text,) in cursor.fetchall():
#                 if text:  # Check if text is not None
#                     # Remove stopwords and tokenize the text
#                     cleaned_text = remove_stopwords(text)
#                     yield utils.simple_preprocess(cleaned_text)

# class Word2VecModel:

#     def __init__(self, db_path, table_name, text_column):
#         self.db_path = db_path
#         self.table_name = table_name
#         self.text_column = text_column

#     def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
#         """Compute cosine similarity between two vectors."""
#         return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

#     def initialize_model(self) -> gensim.models.Word2Vec:
#         """Initialize and train the Word2Vec model with data from SQLite."""
#         sentences = SQLiteCorpus(self.db_path, self.table_name, self.text_column)
#         return gensim.models.Word2Vec(sentences=sentences)

#     def extract_related_terms(self, query_term, drug_list):
#         """Return a pandas dataframe of dictionary terms ranked on cosine similarity"""
#         model = self.initialize_model()
#         data = []
#         for drug in drug_list:
#             row = {
#                 "query_term": query_term,
#                 "drug": drug,
#                 "cosine_similarity_score": self.cosine_similarity(model.wv[query_term], model.wv[drug])
#             }
#             data.append(row)

#         df = pd.DataFrame(data)

#         df = df.sort_values('cosine_similarity_score', ascending=False)

#         return df

# # # Initialize the model (you should do this when starting the application)
# # DB_PATH = 'instance/data.db'
# # TABLE_NAME = 'queryData'  # Replace with your table name
# # TEXT_COLUMN = 'abstract'  # Replace with your column name
# # model = Word2VecModel(DB_PATH, TABLE_NAME, TEXT_COLUMN)

# # model = model.initialize_model()

# # print(model.cosine_similarity(model.wv["chronic"], model.wv["thromboembolic"]))

# # breakpoint()

# from flask import Flask, jsonify
# from gensim import utils
# from gensim.parsing.preprocessing import remove_stopwords
# import gensim.models
# import numpy as np
# import sqlite3
# from typing import List



# class SQLiteCorpus:
#     """An iterator that yields sentences (lists of str) from SQLite database."""
#     def __init__(self, db_path: str, table_name: str, text_column: str):
#         self.db_path = db_path
#         self.table_name = table_name
#         self.text_column = text_column

#     def __iter__(self):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             # Query all text data from the specified table and column
#             cursor.execute(f"SELECT {self.text_column} FROM {self.table_name}")
            
#             for (text,) in cursor.fetchall():
#                 if text:  # Check if text is not None
#                     # Remove stopwords and tokenize the text
#                     cleaned_text = remove_stopwords(text)
#                     yield utils.simple_preprocess(cleaned_text)

# def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
#     """Compute cosine similarity between two vectors."""
#     return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# def initialize_model(db_path: str, table_name: str, text_column: str) -> gensim.models.Word2Vec:
#     """Initialize and train the Word2Vec model with data from SQLite."""
#     sentences = SQLiteCorpus(db_path, table_name, text_column)
#     return gensim.models.Word2Vec(sentences=sentences)

# # Initialize the model (you should do this when starting the application)
# DB_PATH = 'instance/data.db'
# TABLE_NAME = 'queryData'  # Replace with your table name
# TEXT_COLUMN = 'abstract'  # Replace with your column name
# model = initialize_model(DB_PATH, TABLE_NAME, TEXT_COLUMN)
# similarity = cosine_similarity(model.wv["thrombosis"], model.wv["riociguat"])

# print(similarity)

from gensim.parsing.preprocessing import remove_stopwords
from gensim import utils
import gensim.models
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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

def main():
    # Configuration
    connection_string = "postgresql://username:password@localhost:5432/your_database"
    abstract_column_name = "abstract"  # Replace with your actual column name
    
    # Train the model
    model = train_word2vec_model(connection_string, abstract_column_name)
    
    # Analyze specific words
    analyze_word_similarity(model, "pulmonary", "hypertension")
    
    # Print top words
    print_top_words(model)

if __name__ == "__main__":
    main()