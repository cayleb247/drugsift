from flask import Flask, jsonify
from gensim import utils
from gensim.parsing.preprocessing import remove_stopwords
import gensim.models
import numpy as np
import sqlite3
from typing import List
import pandas as pd

app = Flask(__name__)

class SQLiteCorpus:
    """An iterator that yields sentences (lists of str) from SQLite database."""
    def __init__(self, db_path: str, table_name: str, text_column: str):
        self.db_path = db_path
        self.table_name = table_name
        self.text_column = text_column

    def __iter__(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Query all text data from the specified table and column
            cursor.execute(f"SELECT {self.text_column} FROM {self.table_name}")
            
            for (text,) in cursor.fetchall():
                if text:  # Check if text is not None
                    # Remove stopwords and tokenize the text
                    cleaned_text = remove_stopwords(text)
                    yield utils.simple_preprocess(cleaned_text)

class Word2VecModel:

    def __init__(self, db_path, table_name, text_column):
        self.db_path = db_path
        self.table_name = table_name
        self.text_column = text_column

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def initialize_model(self) -> gensim.models.Word2Vec:
        """Initialize and train the Word2Vec model with data from SQLite."""
        sentences = SQLiteCorpus(self.db_path, self.table_name, self.text_column)
        return gensim.models.Word2Vec(sentences=sentences)

    def extract_related_terms(self, query_term, drug_list):
        """Return a pandas dataframe of dictionary terms ranked on cosine similarity"""
        model = self.initialize_model()
        data = []
        for drug in drug_list:
            row = {
                "query_term": query_term,
                "drug": drug,
                "cosine_similarity_score": self.cosine_similarity(model.wv[query_term], model.wv[drug])
            }
            data.append(row)

        df = pd.DataFrame(data)

        df = df.sort_values('cosine_similarity_score', ascending=False)

        return df
    



# Initialize the model (you should do this when starting the application)
DB_PATH = 'instance/data.db'
TABLE_NAME = 'queryData'  # Replace with your table name
TEXT_COLUMN = 'abstract'  # Replace with your column name

sentences = SQLiteCorpus(DB_PATH, TABLE_NAME, TEXT_COLUMN)

print(list(sentences)[-1])

breakpoint()
model = Word2VecModel(DB_PATH, TABLE_NAME, TEXT_COLUMN)

df = model.extract_related_terms("cteph", ["riociguat"])

print(df.head())