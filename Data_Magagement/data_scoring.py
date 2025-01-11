from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import pandas as pd

class tfidfScorer:

    def __init__(self, corpora: list, term_set: set):

        # add set of terms into class
        self.term_set = term_set
        
        # convert list of abstracts into dictionary object
        self.dict = Dictionary(corpora)

        # convert into bag of words (BOWs) form
        self.corpus = [self.dict.doc2bow(abstract) for abstract in corpora]

        self.model = TfidfModel(self.corpus)  # fit model
        
    def aggregate_score(self, term: str):

        term_scores = []

        for doc in self.corpus: # iterate through each BOWs abstract
            tfidf_vector = self.model[doc]
            for term_id, score in tfidf_vector:
                term = self.dict[term_id]
                if term in self.term_set:
                    term_scores[term] = term_scores.get(term, 0) + score

        # Convert the aggregated scores to a pandas DataFrame
        df = pd.DataFrame(list(term_scores.items()), columns=["Term", "Aggregate_Score"])
        df.sort_values(by="Aggregate_Score", ascending=False, inplace=True)  # Sort by score
        df.reset_index(drop=True, inplace=True)
        return df
                

