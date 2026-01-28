import argparse
import os
import shutil
import tempfile

import pandas as pd

from rual.scoring.scorers import SMINA


def _read_smi(path: str) -> pd.DataFrame:
    """Read space-separated SMILES file.

    Expected format per line:
        SMILES name

    If only one column is present, names are generated from row index.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 1:
        raise ValueError(f"No columns found in {path}")

    if df.shape[1] == 1:
        df.columns = ["smi"]
        df["name"] = [str(i) for i in range(len(df))]
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ["smi", "name"]
        df["name"] = df["name"].astype(str)

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run SMINA docking for a .smi file and write scores to .csv.",
        epilog=(
            "Example: rual-smina -i molecules.smi -o scores.csv "
            "--r receptor.pdbqt --l ligand.pdbqt --smina /path/to/smina"
        ),
    )

    parser.add_argument("-i", "--input", required=True, help="Input .smi file (space-separated: SMILES name)")
    parser.add_argument("-o", "--output", default=None, help="Output .csv file (default: <input>.csv)")

    parser.add_argument("--r", required=True, help="Path to receptor .pdbqt")
    parser.add_argument("--l", required=True, help="Path to reference ligand .pdbqt (used for autobox)")
    parser.add_argument("--smina", required=True, help="Path to SMINA executable (or name on PATH)")
    parser.add_argument("--exhaustiveness", type=int, default=1, help="SMINA exhaustiveness (default: 1)")

    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for temporary docking files (default: auto temp dir)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Do not delete workdir after run (only meaningful with --workdir)",
    )

    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    output_path = args.output
    if output_path is None:
        root, _ = os.path.splitext(args.input)
        output_path = root + ".csv"

    smina_exe = args.smina
    if os.path.sep not in smina_exe:
        resolved = shutil.which(smina_exe)
        if resolved is None:
            raise SystemExit(f"Could not find SMINA executable on PATH: {smina_exe}")
        smina_exe = resolved
    else:
        if not os.path.exists(smina_exe):
            raise SystemExit(f"SMINA executable not found: {smina_exe}")

    for p in (args.r, args.l):
        if not os.path.exists(p):
            raise SystemExit(f"File not found: {p}")

    df = _read_smi(args.input)

    scorer = SMINA(
        {
            "r": args.r,
            "l": args.l,
            "smina": smina_exe,
            "exhaustiveness": args.exhaustiveness,
        }
    )

    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="rual_smina_") as tmp:
            df_scored = scorer.score(df[["smi"]].copy(), tmp)
    else:
        os.makedirs(args.workdir, exist_ok=True)
        df_scored = scorer.score(df[["smi"]].copy(), args.workdir)
        if not args.keep_workdir:
            try:
                shutil.rmtree(args.workdir)
            except OSError:
                pass

    df_out = df.copy()
    df_out["score"] = df_scored["score"].to_numpy()
    df_out.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
