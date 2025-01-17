import pandas as pd
from flask import jsonify

def csv2json(path: str):
    '''
    Converts a csv file into a json file to be displayed

    Parameters:
    path (str): path to the csv file

    Returns:
    json: csv file as a json file
    '''
    df = pd.read_csv(path)

    return jsonify(df.to_dict(orient='records'))