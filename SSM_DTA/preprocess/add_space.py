import re
import io
import argparse
from tqdm import tqdm
import multiprocessing

def addspace(pro):
    return ' '.join(list(pro))

def run_add_space(fn, workers, output_fn):
    input_fn = fn
    def lines():
        with io.open(input_fn, 'r', encoding='utf8', newline='\n') as srcf:
            for line in srcf:
                yield line.strip()

    results = []
    total = len(io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines())

    pool = multiprocessing.Pool(workers)
    for res in tqdm(pool.imap(addspace, lines(), chunksize=100000), total=total):
        if res:
            results.append('{}\n'.format(res))

    if output_fn is None:
        output_fn = '{}.pro.addspace'.format(input_fn)
    else:
        output_fn = output_fn
    io.open(output_fn, 'w', encoding='utf8', newline='\n').writelines(results)
    print('{}/{}'.format(len(results), total))


