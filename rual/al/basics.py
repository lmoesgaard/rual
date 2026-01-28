import os
import shutil
import pickle
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

from rual.ml.mltools import get_train_mask


class Utilities():
    """
    Class to handle basic utilities for the RUAL algorithm
    """

    def create_output_dir(self):
        """
        Create output directories for the current round
        """
        if self.round == 1:
            if os.path.exists(self.output) and os.path.isdir(self.output):
                if self.O:
                    print('Overwriting output')
                    shutil.rmtree(self.output)
                    os.mkdir(self.output)
                else:
                    print('Output path found. Not overwriting.')
                    exit()
            else:
                try:
                    os.mkdir(self.output)
                except:
                    print(f'Could not make output directory')
                    exit()
            os.mkdir(os.path.join(self.output, 'train'))
            os.mkdir(os.path.join(self.output, 'test'))
        else:
            # restart from previous round, i.e., load taken molecules and visits
            self.taken = pickle.load(open(os.path.join(self.output, 'taken.pkl'), 'rb'))
            self.db.visits = pickle.load(open(os.path.join(self.output, 'visits.pkl'), 'rb'))

    def create_scoring_dir(self):
        """
        Create working directory for scoring
        """
        self.workdir = os.path.join(self.output, f'round{self.round}')
        os.mkdir(self.workdir)
        scordir = os.path.join(self.workdir, 'scoring')
        os.mkdir(scordir)
        return scordir
    
    def predict(self, id):
        fps = self.db.read_fp_bundle(id)
        a = self.model.predict(fps)
        df = pd.DataFrame({'fileid': id, 'ind': np.arange(a.shape[0]), 'pred': a})
        df["unique_id"] = df["fileid"]*(10**self.base_max_bundle_size) + df["ind"]
        return df
    
    def make_predictions(self, fileids):
        """
        Predict the scores for the molecules in the files specified by fileids
        """
        pred = Parallel(n_jobs=min(len(fileids), self.cpus))(delayed(self.predict)(fid) for fid in fileids)
        #pred = Parallel(n_jobs=1)(delayed(self.predict)(fid) for fid in fileids)
        pred = pd.concat(pred)
        pred = pred[pred.apply(lambda x: x["unique_id"] not in self.taken, axis=1)]
        return pred
    
    def add_smiles_to_df(self, df):
        """
        Grab the SMILES and names for the molecules in the dataframe
        """
        df = df.reset_index(drop=True)
        df['smi'] = None
        df['name'] = None
        for i, subset in df.groupby("fileid"):
            smi = self.db.read_smi_bundle(i)
            inds = subset['ind'].to_numpy()
            df.loc[subset.index, 'smi'] = smi["smi"].to_numpy()[inds]
            df.loc[subset.index, 'name'] = smi["name"].to_numpy()[inds]
        return df

    def combine_fp_files(self, settype):
        """
        Concatenate all fingerprint files in train/test directory and return that data
        """
        return np.concatenate([np.load(os.path.join(self.output, settype, file), allow_pickle=True)['array']
                                for file in os.listdir(os.path.join(self.output, settype)) if file.endswith('.npz')])

    def save_fps(self, df):
        """
        Save the fingerprints and scores to a numpy file
        """
        # if round == 1: split data into trainset and testset
        if self.round == 1: 
            mask = get_train_mask(df, 1-self.test_fraction)
        else: # else: all is saved to train set
            mask = np.ones(len(df)).astype(bool)
        for m, n in zip([mask, ~mask], ['train', 'test']):
            # save trainset to new numpy file
            array = df[m][['fp', 'score']].to_numpy()
            np.savez_compressed(os.path.join(self.output, n, f'round{self.round}'), array=array)

