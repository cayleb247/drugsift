import requests, sys, json

def get_protein_accession(disease_id):
   
  params = {
    "query": f"(cc_disease:{disease_id})",
    "fields": [
      "accession",
      "protein_name"
    ],
    "sort": "accession desc",
    "size": "50"
  }
  headers = {
    "accept": "application/json"
  }
  base_url = "https://rest.uniprot.org/uniprotkb/search"

  response = requests.get(base_url, headers=headers, params=params)
  if not response.ok:
    response.raise_for_status()
    sys.exit()

  data = response.json()

  protein_accessions = []
  for entry in data['results']:
    protein_accessions.append(entry['primaryAccession'])

  return protein_accessions

def get_protein_sequence(accession_id):
    url = f"https://rest.uniprot.org/uniprotkb/{accession_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        # Extract the sequence part (ignoring the header line)
        sequence = "".join(fasta_data.splitlines()[1:])
        return sequence
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")
