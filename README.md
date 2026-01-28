# RUAL: Ramp Up Active Learning

**RUAL** (Ramp Up Active Learning) is a strategy to accelerate molecular docking by combining active learning with a progressive sampling approach. Instead of performing full-database inference in every iteration, RUAL gradually increases the number of molecules evaluated, allowing faster early iterations and more efficient model training.

This ramp-up design balances exploration and exploitation while significantly reducing computational cost—making it practical to apply active learning to ultra-large molecular libraries.

RUAL has been developed to support virtual screening workflows where docking resources are limited but vast chemical space needs to be explored.

## Installation

You can install RUAL using pip:

```sh
pip install -e .
```

Dependencies are listed in `requirements.txt`.

## Running RUAL

RUAL can be used in two ways: via a fully **automated pipeline** for ease of use, or a **manual pipeline** for greater flexibility and customization. Both approaches are demonstrated in the accompanying Jupyter notebooks.

### 1. Automated Pipeline (`automated.ipynb`)

The automated pipeline handles the full RUAL workflow using built-in functions. You define all input parameters in a `@dataclass`, and RUAL handles the rest. This mode is ideal for large-scale screening tasks where the built-in functions are sufficient.

To run, simply define the settings, save the configuration, and execute the pipeline using the `main()` function. 

### 2. Manual Pipeline (`manual.ipynb`)

The manual pipeline provides full control over each component. You can define your own surrogate models, custom scoring functions (e.g. LogP), and control iteration logic manually. This is well suited for experimentation, benchmarking, or integrating alternative docking/scoring strategies.

---

## Configuration Parameters

Below is a summary of key arguments used in both pipelines:

| Parameter         | Type      | Description                                                                 |
|------------------|-----------|-----------------------------------------------------------------------------|
| `O`              | `bool`    | Whether to overwrite existing output                                       |
| `o`              | `str`     | Path to output directory                                                   |
| `database`       | `str`     | Path to database with `smis/` and `fps/` folders                           |
| `config`         | `str`     | Path where the config file will be written                                 |
| `batch_size`     | `int`     | Number of molecules evaluated per round                                    |
| `final_sample`   | `int`     | Number of molecules evaluated in the final iteration                       |
| `iterations`     | `int`     | Total number of active learning iterations                                 |
| `base_bundles`   | `int`     | Number of batches sampled in the first iteration                           |
| `max_bundle_size`| `int`     | Maximum number of molecules in a bundle                                    |
| `rual`           | `bool`    | If `True`, use ramp-up (subsample bundles each round). If `False`, run regular AL (predict all bundles each round) |
| `model_name`     | `str`     | Name of the surrogate model (defined in `ml/models.py`)                    |
| `test_fraction`  | `float`   | Fraction of data reserved as a test set after round 1                      |
| `cpus`           | `int`     | Number of CPUs used per docking round                                      |
| `restart`        | `int`     | Iteration to resume from (set >1 to restart a previous run)                |
| `scorer`         | `dict`    | Dictionary specifying the scoring module and class                         |

---

For complete usage examples, see the [automated.ipynb](automated.ipynb) and [manual.ipynb](manual.ipynb) notebooks in this repository.

## Surrogate Models

RUAL includes built-in surrogate models for predicting docking scores based on molecular fingerprints. These models are lightweight and compatible with scikit-learn. You can easily define your own model classes by following the same interface.

The following surrogate models are available in [`ml/models.py`](ml/models.py):

| Model Name | Description                                                 | Backend          |
|------------|-------------------------------------------------------------|------------------|
| `RF1std`   | A standard Random Forest Regressor with parallel execution  | `RandomForestRegressor` |
| `NN1`      | A simple 2-layer feedforward neural network                 | `MLPRegressor`   |


To use a specific model, simply set the `model_name` argument in your configuration (e.g., `model_name = "NN1"`).

---

You can also define and plug in your own surrogate models by replicating the structure shown in [manual.ipynb](manual.ipynb).

## Scoring Tools

RUAL supports flexible scoring functions used to evaluate molecules during each active learning round. Two scoring tools are included by default, and custom scoring functions can be implemented by following the same interface.

### Built-in Scoring Tools

| Scorer  | Description                                                              | Notes |
|---------|--------------------------------------------------------------------------|-------|
| `LogP`  | Calculates the molecular logP (hydrophobicity) using RDKit               | No additional arguments required |
| `SMINA` | Runs molecular docking using [SMINA](https://github.com/mwojcikowski/smina) | Requires receptor, ligand, and path to SMINA executable |

#### SMINA Arguments

To use the `SMINA` scoring tool, the following arguments **must** be defined in the configuration (structure shown in [automated.ipynb](automated.ipynb)):

- `r`: Path to the receptor `.pdbqt` file
- `l`: Path to the reference ligand `.pdbqt` file
- `smina`: Path to the SMINA executable (e.g., `/path/to/smina.static`)
- *(Optional)* `exhaustiveness`: SMINA docking exhaustiveness (default: 1)

## Building a RUAL-Compatible Database

Before running RUAL, your molecule library must be preprocessed into fingerprint batches. This is done by running the dbbuilder from the command line:

```sh
python3 rual/database/dbbuilder.py -O -i molecules.smi -o ./rual_db -n 10 --cpu 4
```
This script splits a `.smi` file into batches and generates:
- `fps/`: NumPy files with Morgan fingerprints
- `smis/`: Corresponding SMILES and names as Parquet files

### Arguments
| Argument | Description                                                                      |
| -------- | -------------------------------------------------------------------------------- |
| `-i`     | **(Required)** Input `.smi` file (space-separated SMILES and names, no header)   |
| `-o`     | Output directory where `fps/` and `smis/` folders will be created (default: `.`) |
| `-O`     | Overwrite existing `fps/` and `smis/` directories if they already exist          |
| `-n`     | Number of batches to split the data into                                         |
| `--cpu`  | Number of CPU processes to use for parallel processing                           |

The `.smi` file should contain one molecule per line, formatted as:

`SMILES name`
### Example Output Structure
```sh
rual_db/
├── fps/
│   ├── 0.npz
│   ├── 1.npz
│   └── ...
└── smis/
    ├── 0.parquet
    ├── 1.parquet
    └── ...
```