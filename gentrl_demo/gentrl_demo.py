"""
DOCSTRING
"""
import gentrl
import moses
import pandas
import torch

torch.cuda.set_device(0)

class Pretrain:
    """
    DOCSTRING
    """
    def __call__(self):
        df = pandas.read_csv('dataset_v1.csv')
        df = df[df['SPLIT'] == 'train']
        df['plogP'] = df['SMILES'].apply(penalized_logP)
        df.to_csv('train_plogp_plogpm.csv', index=None)
        enc = gentrl.RNNEncoder(latent_size=50)
        dec = gentrl.DilConvDecoder(latent_input_size=50)
        model = gentrl.GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)
        model.cuda();
        md = gentrl.MolecularDataset(
            sources=[{
                'path':'train_plogp_plogpm.csv',
                'smiles': 'SMILES',
                'prob': 1,
                'plogP' : 'plogP',
                }], 
            props=['plogP'])
        train_loader = torch.utils.data.DataLoader(
            md, batch_size=50, shuffle=True, num_workers=1, drop_last=True)
        model.train_as_vaelp(train_loader, lr=1e-4)
        model.save('saved_gentrl')

    def get_num_rings_6(self, mol):
        """
        DOCSTRING
        """
        r = mol.GetRingInfo()
        return len([x for x in r.AtomRings() if len(x) > 6])

    def penalized_logP(self, mol_or_smiles, masked=False, default=-5):
        """
        DOCSTRING
        """
        mol = moses.metrics.utils.get_mol(mol_or_smiles)
        if mol is None:
            return default
        reward = moses.metrics.logP(mol) - moses.metrics.SA(mol) - get_num_rings_6(mol)
        if masked and not moses.metrics.mol_passes_filters(mol):
            return default
        return reward

class TrainRL:
    """
    DOCSTRING
    """
    def __call__(self):
        enc = gentrl.RNNEncoder(latent_size=50)
        dec = gentrl.DilConvDecoder(latent_input_size=50)
        model = gentrl.GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)
        model.cuda();
        model.load('saved_gentrl')
        model.cuda();
        moses.utils.disable_rdkit_log()
        model.train_as_rl(penalized_logP)
        model.save('saved_gentrl_after_rl')

    def get_num_rings_6(mol):
        """
        DOCSTRING
        """
        r = mol.GetRingInfo()
        return len([x for x in r.AtomRings() if len(x) > 6])

    def penalized_logP(mol_or_smiles, masked=False, default=-5):
        """
        DOCSTRING
        """
        mol = moses.metrics.utils.get_mol(mol_or_smiles)
        if mol is None:
            return default
        reward = moses.metrics.logP(mol) - moses.metrics.SA(mol) - get_num_rings_6(mol)
        if masked and not moses.metrics.mol_passes_filters(mol):
            return default
        return reward

if __name__ == '__main__':
    pretrain = Pretrain()
    pretrain()
