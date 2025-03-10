import requests


def getDrugIds(drug_list: list):
    '''
    Use ChEMBL API to get ChEMBL id for each drug of a list

    Parameters:
    drug_list (list): an input list of drugs

    Returns:
    list: a list of entry dictionaries with drug name and ID
    '''

    drug_data = []

    for drug in drug_list:

        # ChEMBL API URL for molecule search
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug}"

        headers = {
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()  # Parse the JSON response

            if data['molecules']:
                first_result = data['molecules'][0] # Get only the first entry's data

                name = first_result.get('pref_name', 'No name available')
                if name == None: # Check if name is unavailable
                    name = f"{drug.upper()}*" # Use input name and mark with asterisk

                chembl_id = first_result['molecule_chembl_id']

                entry = {
                    "name": name,
                    "chembl_id": chembl_id
                }
                drug_data.append(entry)
            else:
                print(f"No compounds found for {drug}")
        else:
            print(f"Failed to fetch data: {response.status_code}")

    return drug_data

def getDrugs(drug_list):

    for drug in drug_list:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{drug['chembl_id']}.json"

        # Send GET request to fetch compound data
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            try:
                smiles = data.get("molecule_structures", {}).get("canonical_smiles", "No SMILES found")

                drug["SMILES"] = smiles

            except:
                print(f"No SMILES string for {drug['name']}")

            try:
                formula = data.get("molecule_properties", {}).get("full_molformula", "No formula found")

                drug["Formula"] = formula

            except:
                print(f"No formula for {drug['name']}")
            
            
            
        else:
            print("Error fetching data from ChEMBL API")
    
    return drug_list


def getDrugData(drug_list: list):
    '''
    Get a list of dictionary entries comprised of drug name, id, and SMILES string
    '''

    drug_ids = getDrugIds(drug_list)

    drug_data = getDrugs(drug_ids)

    return(drug_data)

def check_if_drug(drug_names: list):
    """
    Returns drug in list if drug SMILES string is found
    """
    smiles_results = set([drug["name"].lower() for drug in getDrugData(drug_names)])

    return smiles_results
