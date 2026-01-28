from abc import ABC, abstractmethod
import os
import subprocess

from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP
from rdkit import RDLogger

from rual.utils.utils import generate_random_name, submit_job
from rual.scoring.gen_confs import smi2sdfs


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
        df["score"] = df.apply(lambda x: self.smi2logp(x['smi']), axis=1)
        return df

class SMINA():
    def __init__(self, arguments):
        self.r = arguments['r']
        self.l = arguments['l']
        self.smina = arguments['smina']
        self.exhaustiveness = arguments.get('exhaustiveness', 1)
        pass

    def run_smina_job(self, conf_file):
        base, ext = os.path.splitext(conf_file)
        output = f"{base}_out{ext}"

        cmd = [
            self.smina,
            "-r",
            self.r,
            "-l",
            conf_file,
            "--autobox_ligand",
            self.l,
            "-o",
            output,
            "--cpu",
            "1",
            "--num_modes",
            "1",
            "--exhaustiveness",
            str(self.exhaustiveness),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(output):
            log_path = f"{base}_smina.log"
            try:
                with open(log_path, "w") as f:
                    f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
                    f.write("STDOUT:\n" + (result.stdout or "") + "\n\n")
                    f.write("STDERR:\n" + (result.stderr or "") + "\n")
            except OSError:
                log_path = None

            msg = [
                "SMINA failed to produce an output SDF.",
                f"Return code: {result.returncode}",
                f"Conf file: {conf_file}",
                f"Expected output: {output}",
            ]
            if log_path is not None:
                msg.append(f"See log: {log_path}")
            raise RuntimeError(" ".join(msg))

        return output
    
    def get_scores(self, scores, output):
        if not os.path.exists(output):
            raise RuntimeError(f"SMINA output file not found: {output}")
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
