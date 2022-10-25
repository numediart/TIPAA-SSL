import pickle
import os
import pandas as pd

def load_model(n_components, target_dim, train_set='librispeech'):
    path = './data/models/model_{}_{}_{}.pkl'.format(train_set, n_components, target_dim)
    if os.path.exists(path):
        model = pd.read_pickle(path)
    else:
        print("This model doesn't exist yet.")
        model = None
    return model