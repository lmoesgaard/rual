import sys
import shutil
import argparse
import logging
from typing import Tuple
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


class DBBuilder:
    def __init__(self, smi: str, fps_path: Path, smis_path: Path, num_cpus: int, num_batches: int):
        self.smi = smi
        self.fps_path = fps_path
        self.smis_path = smis_path
        self.num_cpus = num_cpus
        self.num_batches = num_batches

    def make_batch_file(self, i: int) -> None:
        df = {"smi": [], "name": []}
        fps = []
        with open(self.smi) as f:
            for j, line in enumerate(f):
                if j % self.num_batches == i:
                    data = line.strip().split(" ")
                    if len(data) == 2:
                        smi, name = data
                        try:
                            fp, smi = process(smi)
                        except ValueError:
                            logging.error(f"Failed with SMILES: {smi}")
                            continue
                        df["smi"].append(smi)
                        df["name"].append(name)
                        fps.append(fp)
                    else:
                        logging.warning(f"Line did not have two columns: {line}")
        df = pd.DataFrame(df)
        df.to_parquet(self.smis_path / f"{i}.parquet")
        fps = np.array(fps)
        np.savez_compressed(self.fps_path / f"{i}.npz", array=fps)


def create_directories(output_path: Path, allow_overwrite: bool) -> Tuple[Path, Path]:
    fps_path = output_path / 'fps'
    smis_path = output_path / 'smis'

    if not output_path.exists():
        fps_path.mkdir(parents=True)
        smis_path.mkdir(parents=True)
        logging.info(f"Directories created: {fps_path}, {smis_path}")
    else:
        if allow_overwrite:
            if fps_path.exists():
                shutil.rmtree(fps_path)
            if smis_path.exists():
                shutil.rmtree(smis_path)
            fps_path.mkdir(parents=True)
            smis_path.mkdir(parents=True)
            logging.info(f"Directories recreated: {fps_path}, {smis_path}")
        else:
            try:
                fps_path.mkdir(parents=True)
                smis_path.mkdir(parents=True)
            except FileExistsError:
                logging.error(f"Directories existed\nQuitting")
                sys.exit(1)
    return fps_path, smis_path

def process(smi: str) -> Tuple[np.ndarray, str]:
    mol = Chem.MolFromSmiles(smi)  # Convert SMILES to RDKit molecule
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)  # Generate Morgan fingerprint
    array = np.zeros((0,), dtype=np.int8)  # Initialize an empty Numpy array
    DataStructs.ConvertToNumpyArray(fp, array)  # Convert fingerprint to Numpy array
    new_smi = Chem.MolToSmiles(mol)
    return array, new_smi

def main() -> None:
    """
    Main function to process a SMILES file and convert it into batches.

    This function sets up logging, parses command-line arguments, validates the input file,
    creates necessary output directories, and initiates the conversion process using multiple
    processes if specified.

    Command-line arguments:
    -O: Allow overwriting of directories.
    -i: Input space-separated SMILES file without header (required).
    -o: Output destination to create directories fps and smis (default: current directory).
    --cpu: Number of processes the script is allowed to use (default: 1).
    -n: Number of batches to split data into (default: 1).

    Raises:
        SystemExit: If the input file does not exist.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description='Process a SMILES file into a RUAL database (fps/ and smis/ bundles).',
        epilog='Example: python -m rual.database.dbbuilder -O -i ZINC.smi -o ./rual_db -n 5 --cpu 5'
    )
    parser.add_argument('-O', action='store_true', help='Allow overwriting of directories')
    parser.add_argument('-i', type=str, required=True, help='Input space-separated SMILES file without header')
    parser.add_argument('-o', type=str, default=".", help='Output destination to create directories fps and smis')
    parser.add_argument('--cpu', type=int, default=1, help='Number of processes the script is allowed to use')
    parser.add_argument('-n', type=int, default=1, help='Number of batches to split data into')

    args = parser.parse_args()

    input_file = Path(args.i)
    if not input_file.is_file():
        logging.error(f"Error: Input file {args.i} does not exist.")
        sys.exit(1)

    output_path = Path(args.o)
    fps_path, smis_path = create_directories(output_path, args.O)

    num_cpus = args.cpu
    num_batches = args.n
    logging.info(f"Number of processes allowed: {num_cpus}")
    logging.info(f"Number of batches: {num_batches}")

    c = DBBuilder(smi=args.i, fps_path=fps_path, smis_path=smis_path, num_cpus=num_cpus, num_batches=num_batches)
    Parallel(n_jobs=num_cpus)(delayed(c.make_batch_file)(i) for i in range(num_batches))

if __name__ == '__main__':
    main()
