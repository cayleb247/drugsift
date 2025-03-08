import logging
import os
import sys
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
import numpy as np

from fairseq.models.roberta import RobertaModel
from torch.nn.utils.rnn import pad_sequence

from preprocess.add_space import addspace
from preprocess.canonicalize import canonicalize
from preprocess.tokenize_re import smi_tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_inference(checkpoint, data_bin, input_mol_fn, input_pro_fn, output_fn, batch_size, input_label_fn=None, mode='predict'):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    roberta = RobertaModel.from_pretrained(
        os.path.split(checkpoint)[0],
        checkpoint_file=os.path.split(checkpoint)[1],
        data_name_or_path=data_bin,
    )
    roberta.to('cpu')
    roberta.eval()
    bsz = batch_size


    total = len(open(input_mol_fn, 'r').readlines())
    pbar = tqdm(total=total, desc='Predicting')
    
    with open(f'{input_mol_fn}', 'r') as mol_in, open(f'{input_pro_fn}', 'r') as pro_in, open(output_fn, 'w') as out_f:
        batch_mol_buf = []
        batch_pro_buf = []
        print(mol_in)
        print(type(mol_in))
        print(list(enumerate(zip(mol_in, pro_in))))
        breakpoint()
        for i, mol_pro in enumerate(zip(mol_in, pro_in)):
            mol, pro = mol_pro
            if ((i+1) % bsz == 0) or ((i+1) == total):
                tmp_mol, tmp_pro = roberta.myencode_separate(mol.strip(), pro.strip())
                batch_mol_buf.append(tmp_mol[:512])
                batch_pro_buf.append(tmp_pro[:1024])
                tokens_0 = pad_sequence(batch_mol_buf, batch_first=True, padding_value=1)
                tokens_1 = pad_sequence(batch_pro_buf, batch_first=True, padding_value=1)
                predictions = roberta.myextract_features_separate(tokens_0, tokens_1)
                for result in predictions:
                    out_f.write(f'{str(result.item())}\n')
                batch_mol_buf.clear()
                batch_pro_buf.clear()
                pbar.update(1)
            else:
                tmp_mol, tmp_pro = roberta.myencode_separate(mol.strip(), pro.strip())
                batch_mol_buf.append(tmp_mol[:512])
                batch_pro_buf.append(tmp_pro[:1024])
                pbar.update(1)
                continue
    
    pbar.close()
    if mode == 'eval':
        assert input_label_fn is not None
        pred = [float(line.strip()) for line in open(output_fn, 'r').readlines()]
        gold = [float(line.strip()) for line in open(input_label_fn, 'r').readlines()]
        print('MSE:', mean_squared_error(gold, pred))
        print('RMSE:', np.sqrt(mean_squared_error(gold, pred))) 
        print('Pearson:', pearsonr(gold, pred))
        print('C-index:', concordance_index(gold, pred))


# run_inference("Models/bindingdb_Ki_SSM.pt", "SSM_DTA/dict", "Models/model_data/input.mol.can.re", "Models/model_data/input.pro.addspace", "Models/model_data/input2.pred.txt", 8)
# breakpoint()


def run_inference_python(checkpoint, data_bin, mol_in, pro_in):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    roberta = RobertaModel.from_pretrained(
        os.path.split(checkpoint)[0],
        checkpoint_file=os.path.split(checkpoint)[1],
        data_name_or_path=data_bin,
    )
    roberta.to('cpu')
    roberta.eval()

    # make batch size one always
    bsz = 1

    # canonicalize and tokenize SMILES string, add space to amino acid seq
    mol_in = canonicalize(mol_in)
    mol_in = smi_tokenizer(mol_in)
    pro_in = addspace(pro_in)

    # since we are not batch processing, the number of compounds will always be one
    total = 1
    pbar = tqdm(total=total, desc='Predicting')
    
    batch_mol_buf = []
    batch_pro_buf = []

    output = []

    for i, mol_pro in enumerate(zip([mol_in], [pro_in])):
        mol, pro = mol_pro
        if ((i+1) % bsz == 0) or ((i+1) == total):
            tmp_mol, tmp_pro = roberta.myencode_separate(mol.strip(), pro.strip())
            batch_mol_buf.append(tmp_mol[:512])
            batch_pro_buf.append(tmp_pro[:1024])
            tokens_0 = pad_sequence(batch_mol_buf, batch_first=True, padding_value=1)
            tokens_1 = pad_sequence(batch_pro_buf, batch_first=True, padding_value=1)
            predictions = roberta.myextract_features_separate(tokens_0, tokens_1)
            for result in predictions:
                output.append(str(result.item()))
                batch_mol_buf.clear()
                batch_pro_buf.clear()
                pbar.update(1)
        else:
            tmp_mol, tmp_pro = roberta.myencode_separate(mol.strip(), pro.strip())
            batch_mol_buf.append(tmp_mol[:512])
            batch_pro_buf.append(tmp_pro[:1024])
            pbar.update(1)
            continue
    
    pbar.close()

    # output list will always contain one prediction, so we use the first index
    return output[0]

class SSM_DTA:
    def __init__(self, checkpoint, data_bin):
        logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
        )
        self.roberta = RobertaModel.from_pretrained(
            os.path.split(checkpoint)[0],
            checkpoint_file=os.path.split(checkpoint)[1],
            data_name_or_path=data_bin,
        )
        self.roberta.to('cpu')
        self.roberta.eval()

        # make batch size one always
        self.bsz = 1

    def run_inference(self, mol_in, pro_in):

        # canonicalize and tokenize SMILES string, add space to amino acid seq
        mol_in = canonicalize(mol_in)
        mol_in = smi_tokenizer(mol_in)
        pro_in = addspace(pro_in)

        # since we are not batch processing, the number of compounds will always be one
        total = 1
        pbar = tqdm(total=total, desc='Predicting')
        
        batch_mol_buf = []
        batch_pro_buf = []

        output = []

        for i, mol_pro in enumerate(zip([mol_in], [pro_in])):
            mol, pro = mol_pro
            if ((i+1) % self.bsz == 0) or ((i+1) == total):
                tmp_mol, tmp_pro = self.roberta.myencode_separate(mol.strip(), pro.strip())
                batch_mol_buf.append(tmp_mol[:512])
                batch_pro_buf.append(tmp_pro[:1024])
                tokens_0 = pad_sequence(batch_mol_buf, batch_first=True, padding_value=1)
                tokens_1 = pad_sequence(batch_pro_buf, batch_first=True, padding_value=1)
                predictions = self.roberta.myextract_features_separate(tokens_0, tokens_1)
                for result in predictions:
                    output.append(str(result.item()))
                    batch_mol_buf.clear()
                    batch_pro_buf.clear()
                    pbar.update(1)
            else:
                tmp_mol, tmp_pro = self.roberta.myencode_separate(mol.strip(), pro.strip())
                batch_mol_buf.append(tmp_mol[:512])
                batch_pro_buf.append(tmp_pro[:1024])
                pbar.update(1)
                continue
        
        pbar.close()

        # output list will always contain one prediction, so we use the first index

        return 10 ** (9 - float(output[0]))

# print(run_inference_python("Models/bindingdb_Ki_SSM.pt", "SSM_DTA/dict", "CN(C1=C(N=C(N=C1N)C2=NN(C3=C2C=CC=N3)CC4=CC=CC=C4F)N)C(=O)OC", "CVMYKDMQVTNVCYRICDWYSSMFVMQVIGFMMDHFMDRFGVLNDPIVWGCPQAERKYDVQFRQFRLKPNLGFLMQFVNYTFEPSKEAMWGLWIPQYFGCWGCLIFWCYINPENMFCLSTWPMYLYIMVFQLIIELFAWIAITCGELVSNTRFWFYFPNWKAEQFKIYAQKSSPNEFVVKYHHYTQANEAMVPQGTWYGAQYEEDASMQYRWQVEHTTWNWFQSWSAGSRTDLCVSCFIAHSSEPKRTFDVGTASLLFNGDDVQPIIFCDKWDAYGLIQQPYAAVKWCVIWNHKYFWAPQCQRQCQMCYKRASDSHPRWFCQYFSDAWKTDDPMCKPDGKGCQLWHGWMKEQSAPRLINAYVHEFWADAGAKPEQRKLLPAQWRKYRPQYHQLVSWLNPLRRKAKEISSVRCCGYSMQDTVCNAECWDLPKPINTIYWLWECEWSETTLKWAEVVWNTNMMTTRIISWIKYESACHVMYFMDYNTGVMHEDLCLWFGNMVNTFEWEMYKKPIFIMNKTIVQQDCHLPYRHRWVDINNNGIAHAFSAFWEIACKQVESHMVGNDEQIAHEAKEVLPAAAWLFTRMQRILPIAIMVIWREAAALYHGFGKIEYVKLSLFPYAPVHTGMQTFQGEVCRWTNADYRKRTQQIKLWHWRIHSRLPTPITKQRGDIPYYIHDAGKPRWVICHKAPNGPFRFFEHAEVSRHDPAQRLAEEYIGMINTMFPPSDGMVYLGKGYDVREPKKNICCSWFKHYWNQLMHMRVIHWYWGDNYIYCMVTPHLAHCCMNTVRLSGRERVPTWGNVNELHTDGTCGATWNVGLLWDPVYLDVILNISNFPFSVRIMQNVLDTCAPWSQDNKMMWPGRSSRMYFGPKNYEMIEENPESAEPDNPTVGTYMHLVPVKTCPEHRMSLYSEQKTFNESMEDLKQLRMFVMKSIYGFGIATRQQVNWQTLWQGDPQYRLGWLATENWPNNEKMQLNHETVMWSNGAAWGWNGVSPEWKYY"))

# predictor = SSM_DTA("Models/bindingdb_Ki_SSM.pt", "SSM_DTA/dict")

# print(predictor.run_inference("CCC(Cc1ccccc1)c1cc(O)c(C(CC)c2ccccc2)c(=O)o1", "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"))
