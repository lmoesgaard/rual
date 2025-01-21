import argparse
import os
import sys
import shutil
import pandas as pd
import importlib

from database import Database
from ml.mltools import get_model

def parse_input():
    parser = argparse.ArgumentParser(description='Read input flags')
    parser.add_argument('-r', type=str, required=True, help='Path to the receptor file (.pdbqt)')
    parser.add_argument('-l', type=str, required=True, help='Path to the xtal file (.pdbqt)')
    parser.add_argument('-o', type=str, required=True, help='Path to the desired output directory')
    parser.add_argument('--smina', type=str, required=True, help='Path to smina executable')
    parser.add_argument('--database', type=str, required=True, help='Path to database')
    parser.add_argument('--O', action='store_true', help='Permission to overwrite', required=False)
    parser.add_argument('--final_sample', type=int, default= 2000000,
                        help='Size of the final sample', required=False)
    parser.add_argument('--batch_size', type=int, default= 10000,
                        help='Number of molecules to dock at each iteration', required=False)
    parser.add_argument('--base_bundles', type=int, default=1,
                        help='Number of bundles evaluated in first iteration', required=False)
    parser.add_argument('--model_name', type=str, default='NN1',
                        help='Name of ML model to use for active learning', required=False)
    parser.add_argument('--cpus', type=int, default=10,
                        help='Number of CPUs to use', required=False)
    parser.add_argument('--iterations', type=int, default=2,
                        help='Number of AL iteration', required=False)
    parser.add_argument('--restart', type=int, default=1,
            help='AL iteration to start from', required=False)
    args = parser.parse_args()
    return args

class test_input():
    def __init__(self):
        self.r = 'test_r.pdbqt'
        self.l = 'test_l.pdbqt'
        self.o = '/Users/moesgaard/scratch/rual'
        self.smina = '/Users/moesgaard/Code/smina.static'
        self.database = '/Users/moesgaard/scratch/rual_db'
        self.O = True
        self.final_sample = 2000000
        self.batch_size = 10000
        self.base_bundles = 1
        self.model_name = 'NN1'
        self.cpus = 10
        self.iterations = 5
        self.restart = 1

class RUAL():
    def __init__(self):
        #args = parse_input()
        args = test_input()

        self.smina = args.smina
        self.receptor = args.r
        self.xtal = args.l
        self.output = args.o
        self.batch_size = args.batch_size
        self.workdir = args.o
        self.cpus = args.cpus
        self.iterations = int(args.iterations)
        self.final_sample = int(args.final_sample)
        self.base_bundles = int(args.base_bundles)
        self.round = args.restart

        self.taken = set()
        self.final = False

        self.model = get_model(args.model_name)
        self.db = Database(args.database, args.base_bundles)
        self.create_output_dir()

    def create_output_dir(self):
        if os.path.exists(self.output) and os.path.isdir(self.output):
            if self.O and self.round == 1:
                print('Overwriting output')
                shutil.rmtree(self.output)
                os.mkdir(self.output)
            elif self.O and self.round == 1:
                print('You cannot overwrite and restart at the same time')
                exit()
            elif self.round == 1:
                print('Output path found. Not overwriting.')
                exit()
        else:
            try:
                os.mkdir(self.output)
            except:
                print(f'Could not make output directory')
                exit()

        if self.round == 1:
            os.mkdir(os.path.join(self.output, 'train'))
            os.mkdir(os.path.join(self.output, 'test'))
        else:
            df = pd.read_parquet(os.path.join(self.output, "output.parquet"))
            self.taken = set(df.id)
            self.visits[df.file.unique() - 1] += 1

    def create_docking_dir(self):
        self.workdir = os.path.join(self.output, f'round{self.round}')
        os.mkdir(self.workdir)
        dockdir = os.path.join(self.workdir, 'docking')
        os.mkdir(dockdir)
        return dockdir
    

def get_model(modelname):
    module = importlib.import_module("ml.models")
    try:
        return getattr(module, modelname)
    except:
        print(f"Could not find model: {modelname}")
        sys.exit()
        