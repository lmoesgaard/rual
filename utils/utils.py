from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pandas as pd
import subprocess
import pickle
import json
import os


def smi2array(smi):
    mol = Chem.MolFromSmiles(smi)  # Convert SMILES to RDKit molecule
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)  # Generate Morgan fingerprint
    array = np.zeros((0,), dtype=np.int8)  # Initialize an empty Numpy array
    DataStructs.ConvertToNumpyArray(fp, array)  # Convert fingerprint to Numpy array
    return array

def get_tanimoto(ref, fps):
    C = (ref * fps).sum(axis=1)  # Calculate intersection between reference and fingerprints
    D = (ref + fps).sum(axis=1)  # Calculate union of reference and fingerprints
    sim = (C / (D - C))  # Calculate Tanimoto similarity
    return sim

def smi2df(smifile):
    df = pd.read_csv(smifile, sep=' ', header=None)  # Read SMILES file
    df.columns = ['smi', 'ids']  # Set column names
    return df

def add_fingerprint_to_dataframe(df):
    df['fp'] = df.apply(lambda x: smi2array(x['smi']), axis=1)
    return df

def df2smi(df, smipath):
    df[['smi', 'name']].to_csv(smipath, sep=' ', header=None, index=None)

def submit_job(job):
    subprocess.run(job, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def split_df(df, n):
    chunk_size = len(df) // n
    remainder = len(df) % n
    return [df.iloc[i*chunk_size + min(i, remainder):(i+1)*chunk_size + min(i+1, remainder)] for i in range(n)]

def pickle_data(dir, data, filename):
    with open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(data, f)

def save_config(settings):
    json_file = settings.config
    with open(json_file, 'w') as f:
        json.dump(settings.__dict__, f, indent=4, sort_keys=True)
