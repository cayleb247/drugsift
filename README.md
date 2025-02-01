### DrugSift

DrugSift is a web application that aims to combine the latest tools in deep learning and biomedical knowledge into an integrated pipeline for drug discovery and repurposing.

## How it works

# 1. NLP Pipeline for Abstract
- Abstracts are taken from PubMed
- Texts are preprocessed
- Drug compounds and clinical feature terms are extracted

# 2. Deep Learning Validation
- Graph Neural Networks are used to validate the binding affinity between drugs and disease-associated proteins
- Takes the ligands as SMILES strings and the proteins as amino acid sequences

# 3. Results and Visualizations
- Drug candidates are further screened against Lipinski's rule of 5
- Radar graphs, drug structure, and other metrics are displayed

NOTE: This project is still a work in progress
