from Bio import Entrez, Medline
import subprocess
from io import StringIO
from multiprocessing import Pool
import pandas as pd
import re

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

def extract_year(text):
    pattern = r'\b(19|20)\d{2}\b'  # Matches years from 1900-2099
    match = re.search(pattern, text)
    return match.group() if match else ""

def get_queried_abstracts(query: str):
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
            "search-query": query,
            "abstract": record.get("AB", ''),
            "year-published": extract_year(record.get("DP", ''))
        }
        articles.append(article)
    
    return pd.DataFrame(articles)