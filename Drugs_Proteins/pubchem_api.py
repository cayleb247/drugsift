import requests

def get_smiles_from_pubchem(drug_names):
    smiles_dict = {}

    for drug in drug_names:
        # PubChem API endpoint for searching compounds by name
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug}/property/CanonicalSMILES/JSON"

        try:
            response = requests.get(search_url)
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                # Extract SMILES if available
                if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                    smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES', 'No SMILES found')
                    smiles_dict[drug] = smiles
                else:
                    smiles_dict[drug] = "No SMILES found"

        except Exception as e:
            smiles_dict[drug] = f"Error: {str(e)}"

    return smiles_dict

def check_if_drug(drug_names):
    """
    Returns drug in list if drug SMILES string is found
    """
    smiles_results = set(get_smiles_from_pubchem(drug_names).keys())

    return smiles_results
