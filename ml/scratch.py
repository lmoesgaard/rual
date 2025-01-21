
        bestmol = self.best_from_multiple_dfs(prospects, final)  # make predictions on the fingerprint bundles of 'prospects'
        if final:
            bestmol = bestmol.sample(self.final_subsample)
        bestmol = self.extract_best(bestmol)  # make the dataframe more informative (FP, index, locations)


best_df = None

        prediction = self.predict_db(i)  # Make prediction on ith bundle
        best_df = pd.concat([prediction, best_df])  # Add prediction to collected DataFrame
        best_df = best_df.sort_values('score') # Sort by score
        best_df = best_df.head(min([len(best_df), bsize]))  # Keep only the best ones
    return best_df


def predict_db(self, i):
    pred = self.model.predict(testX) # make prediction using ML model
    pred_df = pd.DataFrame(pred, columns=['score']) # turn prediction into dataframe
    pred_df['file'] = i # save the filename for later, so that molecules can be picked out
    pred_df = pred_df.reset_index() # also get the indexing for later
    return pred_df

def extract_best(self, df):
    new = None
    for file in df['file'].unique(): # for each bundle with good molecules
        mask = df[df['file'] == file]['index'].to_numpy() # get the indices of good molecules in bundle
        current = pd.read_parquet(os.path.join(self.db, f'smi_prot/{file}.parquet')).iloc[mask] # read dataframes
        current['fp'] = list(np.load(os.path.join(self.db, f'fps_prot/{file}.npz'))['array'][mask]) # add fps
        current['file'] = file # save name of bundle for later
        current['index'] = mask # save indices for later
        new = pd.concat([new, current]) # add to parent dataframe
    new = new[~new['id'].isin(self.taken)] # remove molecules that have already been docked
    return new