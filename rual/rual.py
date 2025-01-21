import os
import shutil
import sys
import pandas as pd
import numpy as np
import importlib
from joblib import Parallel, delayed

from utils.utils import df2smi, split_df, pickle_data, add_fingerprint_to_dataframe
from database.database import Database
from rual.basics import Utilities
from ml.mltools import get_model


def get_model(modelname):
    module = importlib.import_module("ml.models")
    try:
        return getattr(module, modelname)()
    except:
        print(f"Could not find model: {modelname}")
        sys.exit()

class RUAL(Utilities):
    def __init__(self, args):
        self.O = bool(args['O'])
        self.output = args['o']
        self.batch_size = int(args['batch_size'])
        self.cpus = int(args['cpus'])
        self.iterations = int(args['iterations'])
        self.final_sample = int(args['final_sample'])
        self.base_bundles = int(args['base_bundles'])
        self.max_bundle_size = int(args['max_bundle_size'])
        self.base_max_bundle_size = np.ceil(np.log10(self.max_bundle_size)
                                            ).astype(int)
        if 'restart' in args:
            self.round = int(args['restart'])
        else:
            self.round = 1
        self.test_fraction = args['test_fraction']

        self.taken = set()
        self.final = False
        
        self.scorer = None
        self.model = get_model(args["model_name"])
        self.db = Database(args["database"], self.base_bundles, self.max_bundle_size)
        self.create_output_dir()
        

    def new_round(self):
        # check if it's the final round
        if self.round == self.iterations:
            self.final = True

        # create new directory for docking
        dockdir = self.create_docking_dir()

        # select fileids from database
        fileids = self.db.query_db(round=self.round, final=self.final)
        
        # run predictions using ML model on selected files and remove those that are already docked
        pred = self.make_predictions(fileids)

        # extract best molecules from predictions, store round number, and update taken set
        bsize = self.batch_size
        if self.final:
            bsize = self.final_sample
        bestmol = pred.sort_values('pred', ascending=False).head(bsize)
        bestmol['round'] = self.round
        self.taken.update(list(bestmol['unique_id']))

        # add smiles and names to df and save .smi file
        bestmol = self.add_smiles_to_df(bestmol)
        df2smi(bestmol, os.path.join(self.workdir, f'molecules_r{self.round}.smi'))

        # run docking on multiple cores:
        bestmol = Parallel(n_jobs=self.cpus)(delayed(self.scorer.score)(subset, dockdir) for subset in split_df(bestmol, self.cpus))
        bestmol = pd.concat(bestmol)
        shutil.rmtree(dockdir)

        # save df-data as parquet file
        bestmol.to_parquet(os.path.join(self.output, f'output_r{self.round}.parquet'))

        # make train set, test set
        bestmol = add_fingerprint_to_dataframe(bestmol)
        self.save_fps(bestmol)

        # train model if it's not the final round
        if not self.final:
            X, y = self.combine_fp_files('train').T # get data
            self.model.model.fit(list(X), y) # train model

        # pickle visits and taken molecules
        pickle_data(self.output, self.db.visits, 'visits.pkl')
        pickle_data(self.output, self.taken, 'taken.pkl')

        # update status
        print(f'Round {self.round} done')
        self.round += 1
