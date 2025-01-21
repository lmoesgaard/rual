from abc import ABC, abstractmethod

from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP


class Scorer(ABC):
    @abstractmethod
    def __init__(self, arguments):
        pass

    def score(self, df, workdir):
        df["score"] = 0
        pass


class LogP():
    def __init__(self, arguments):
        pass

    def smi2logp(self, smi):
        mol = Chem.MolFromSmiles(smi)
        logp = MolLogP(mol)
        return logp

    def score(self, df, workdir):
        """
        Run docking on dataframe with a smi and a uniqueid column
        """
        df["score"] = df.apply(lambda x: self.smi2logp(x['smi']), axis=1)
        return df
