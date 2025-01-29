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
            else:
                smiles_dict[drug] = f"Error: {response.status_code}"

        except Exception as e:
            smiles_dict[drug] = f"Error: {str(e)}"

    return smiles_dict

# List of drugs
drug_list = ["sudafed", "visine", "riociguat", "oxycodone"]
smiles_results = get_smiles_from_pubchem(drug_list)

# Print results
for drug, smiles in smiles_results.items():
    print(f"{drug}: {smiles}")
