from Bio import Entrez, Medline
import subprocess
from io import StringIO
from multiprocessing import Pool
import pandas as pd

def get_pubmed_count(query, email=None):
    """
    Get the number of PubMed articles matching a search query using Biopython.
    
    Parameters:
    query (str): The search term(s) to look for in PubMed
    email (str): Your email address (required by NCBI)
    
    Returns:
    int: Number of matching articles
    """
    # Set your email address (required by NCBI)
    if email:
        Entrez.email = email
    
    try:
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, rettype="count")
        record = Entrez.read(handle)
        handle.close()
        
        # Get the count of results
        count = int(record["Count"])
        
        return count
    
    except Exception as e:
        return None


def getQueriedAbstracts(query: str):
    '''
    Using NCBI's EDirect Unix command line application, get all abstracts of the user's query.

    Parameters:
    query (str): User's disease query

    Returns:
    df: a pandas df containing all abstracts of given query
    '''

    esearch_cmd = f"esearch -db pubmed -query '{query}'"
    efetch_cmd = f"efetch -format medline"

    command = f"{esearch_cmd} | {efetch_cmd}" # join the two commands with a pipe

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    records = Medline.parse(StringIO(result.stdout))

    articles = []

    for record in records:
        article = {
            "abstract": record.get("AB", "")
        }
        articles.append(article)
    
    return pd.DataFrame(articles)


df = getQueriedAbstracts("venous thrombosis")

print(df.head())
print(df.isna().any().any())
print(df.shape)
