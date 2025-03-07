import re
import io
import argparse
from tqdm import tqdm
import multiprocessing


def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub('\s+', '', smi) == ''.join(tokens)
    except:
        return ''

    return ' '.join(tokens)

def run_tokenization(fn, workers, output_fn):
    input_fn = fn
    def lines():
        with io.open(input_fn, 'r', encoding='utf8', newline='\n') as srcf:
            for line in srcf:
                yield line.strip()

    results = []
    total = len(io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines())

    pool = multiprocessing.Pool(workers)
    for res in tqdm(pool.imap(smi_tokenizer, lines(), chunksize=100000), total=total):
        if res:
            results.append('{}\n'.format(res))

    if output_fn is None:
        output_fn = '{}.bpe'.format(input_fn)
    else:
        output_fn = output_fn
    io.open(output_fn, 'w', encoding='utf8', newline='\n').writelines(results)
    print('{}/{}'.format(len(results), total))

