import argparse
import os
import shutil
import tempfile

import pandas as pd
from joblib import Parallel, delayed

from rual.scoring.scorers import SMINA
from rual.utils.utils import split_df


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


def _score_chunk(chunk: pd.DataFrame, scorer_args: dict, run_workdir: str, keep_workdir: bool) -> pd.DataFrame:
    """Score a chunk of SMILES.

    Returns a dataframe with columns: __rowid, score.
    """
    os.makedirs(run_workdir, exist_ok=True)
    chunk_dir = tempfile.mkdtemp(prefix="chunk_", dir=run_workdir)
    scorer = SMINA(scorer_args)

    try:
        df_scored = scorer.score(chunk[["smi"]].copy(), chunk_dir)
        out = pd.DataFrame(
            {
                "__rowid": chunk["__rowid"].to_numpy(),
                "score": df_scored["score"].to_numpy(),
            }
        )
    except Exception:
        # Keep the chunk directory on failure for debugging.
        raise
    else:
        if not keep_workdir:
            try:
                shutil.rmtree(chunk_dir)
            except OSError:
                pass
        return out


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
        "--n-proc",
        "--cpus",
        dest="n_proc",
        type=int,
        default=1,
        help="Number of parallel SMINA processes (default: 1)",
    )

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
    df = df.reset_index(drop=True)
    df["__rowid"] = range(len(df))

    scorer_args = {
        "r": args.r,
        "l": args.l,
        "smina": smina_exe,
        "exhaustiveness": args.exhaustiveness,
    }

    # Use a per-run subdirectory to avoid deleting user-owned directories.
    created_base_tmp = False
    base_workdir = args.workdir
    if base_workdir is None:
        base_workdir = tempfile.mkdtemp(prefix="rual_smina_")
        created_base_tmp = True
    else:
        os.makedirs(base_workdir, exist_ok=True)

    run_workdir = tempfile.mkdtemp(prefix="run_", dir=base_workdir)

    n_proc = max(int(args.n_proc), 1)
    n_proc = min(n_proc, len(df)) if len(df) > 0 else 1

    chunks = split_df(df[["smi", "__rowid"]], n_proc)

    try:
        if n_proc == 1:
            scored_parts = [_score_chunk(chunks[0], scorer_args, run_workdir, keep_workdir=args.keep_workdir)]
        else:
            scored_parts = Parallel(n_jobs=n_proc)(
                delayed(_score_chunk)(chunk, scorer_args, run_workdir, keep_workdir=args.keep_workdir)
                for chunk in chunks
            )

        scores_df = pd.concat(scored_parts, ignore_index=True)
        df_out = df.merge(scores_df, on="__rowid", how="left")
        df_out = df_out.drop(columns=["__rowid"])
        df_out.to_csv(output_path, index=False)
    finally:
        # Clean up run directory if requested; always keep it on failure for debugging.
        # (If an exception escapes, Python will skip this removal, leaving run_workdir intact.)
        if not args.keep_workdir:
            try:
                shutil.rmtree(run_workdir)
            except OSError:
                pass
        if created_base_tmp and not args.keep_workdir:
            try:
                shutil.rmtree(base_workdir)
            except OSError:
                pass


if __name__ == "__main__":
    main()
