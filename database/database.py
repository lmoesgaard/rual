import numpy as np
import pandas as pd
import os

class Database():
    def __init__(self, path, base_bundles, final_bundles=np.inf):
        self.path = path
        self.smi_files = {int(f.split(".")[0]): f for f in os.listdir(os.path.join(path, 'smis')) if f.endswith('.parquet')}
        self.fp_files = {int(f.split(".")[0]): f for f in os.listdir(os.path.join(path, 'fps')) if f.endswith('.npz')}
        assert len(self.smi_files) == len(self.fp_files), 'Number of SMILES and fingerprints do not match'
        self.n_bundles = len(self.smi_files)
        self.visits = np.zeros(self.n_bundles)
        self.min_bundles = base_bundles
        self.final_bundles = final_bundles

    def query_db(self, round, final=False):
        p = self.get_props()  # probability of the bundle is proportional to how few times it has been selected
        if final:
            prospects = np.arange(min(self.n_bundles, self.final_bundles))
        else:
            prospects = np.random.choice(np.arange(self.n_bundles), # n bundles available
                                         size= self.min_bundles * 2 ** (round - 1),
                                         replace=False, p=p)
            self.visits[prospects] += 1 # mark that the selected bundles have been visited this round
        return prospects

    def get_props(self):
        p = self.visits.max() - self.visits + 0.01
        p = p / p.sum()
        return p
    
    def read_smi_bundle(self, i):
        smi = pd.read_parquet(os.path.join(self.path, 'smis', self.smi_files[i]))
        return smi
    
    def read_fp_bundle(self, i):
        fp = np.load(os.path.join(self.path, 'fps', self.fp_files[i]))['array']
        return fp
    