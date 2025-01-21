from abc import ABC, abstractmethod
import os

from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP
from rdkit import RDLogger

from utils.utils import generate_random_name, submit_job
from scoring.gen_confs import smi2sdfs


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

class SMINA():
    def __init__(self, arguments):
        self.r = arguments['r']
        self.l = arguments['l']
        self.smina = arguments['smina']
        if "exhaustiveness" in arguments:
            self.exhaustiveness = arguments['exhaustiveness']
        else:
            self.exhaustiveness = 1
        pass

    def run_smina_job(self, conf_file):
        output = conf_file.replace(".", "_out.")
        job = f'{self.smina} -r {self.r} -l {conf_file} \
                --autobox_ligand {self.l} -o {output} \
                --cpu 1 --num_modes 1 --exhaustiveness {self.exhaustiveness}'
        submit_job(job)
        return output
    
    def get_scores(self, scores, output):
        RDLogger.DisableLog('rdApp.warning')
        supplier = Chem.SDMolSupplier(output)
        for mol in supplier:
            score = mol.GetPropsAsDict()['minimizedAffinity']
            score = max(-score, 0)
            molid = int(mol.GetProp('_Name'))
            scores[molid] = score
        return scores

    def score(self, df, workdir):
        """
        Run docking using SMINA
        """
        name = generate_random_name()
        conf_file = os.path.join(workdir, f'{name}.sdf')
        smi2sdfs(smis=df['smi'], filename=conf_file)
        output = self.run_smina_job(conf_file)
        scores = [0] * len(df)
        scores = self.get_scores(scores=scores, output=output)
        df["score"] = scores
        return df
    