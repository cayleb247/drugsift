import re
import io
import argparse
from tqdm import tqdm
import multiprocessing
from rdkit import Chem


def rm_map_number(smiles):
    t = re.sub(':\d*', '', smiles)
    return t


def canonicalize(smiles):
    try:
        smiles = rm_map_number(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        else:
            return Chem.MolToSmiles(mol)
    except:
        return None


def run_canonicalization(fn, workers, output_fn, keep_atommapnum=False):
    input_fn = fn
    def lines():
        with io.open(input_fn, 'r', encoding='utf8', newline='\n') as srcf:
            for line in srcf.readlines():
                yield line.strip(), keep_atommapnum

    results = []
    total = len(io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines())

    pool = multiprocessing.Pool(workers)
    for res in tqdm(pool.imap(canonicalize, lines(), chunksize=100000), total=total):
        if res is not None:
            results.append('{}\n'.format(res))

    if output_fn is None:
        output_fn = '{}.can'.format(input_fn)
    else:
        output_fn = output_fn
    io.open(output_fn, 'w', encoding='utf8', newline='\n').writelines(results)
    print('{}/{}'.format(len(results), total))