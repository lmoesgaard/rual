import sys
import numpy as np
import importlib


def get_model(modelname):
    """
    import SciKit model from ml.models
    """
    module = importlib.import_module("ml.models")
    try:
        return getattr(module, modelname)()
    except:
        print(f"Could not find model: {modelname}")
        sys.exit()


def get_train_mask(dataset, split_ratio):
    """
    Generate a boolean mask for splitting a dataset into a train and a test set
    """
    total_samples = len(dataset) # need the length of dataset
    num_train_samples = int(split_ratio * total_samples) # find the number of samples to put in trainset

    train_mask = np.array([True] * num_train_samples # add number all trainmasks to list
                          + [False] * (total_samples - num_train_samples))  # add all testmasks to list
    np.random.shuffle(train_mask) # shuffle train and test values
    
    return train_mask
