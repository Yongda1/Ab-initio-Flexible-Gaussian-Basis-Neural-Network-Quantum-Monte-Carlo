import ml_collections
from ml_collections import config_dict

def default() -> ml_collections.ConfigDict:

    cfg = ml_collections.ConfigDict({
        'batch_size': 128,
        'pos': [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
        'charges': [6.],
        'spins': [1, -1, 1, -1, 1, -1],
        'atoms': [[0.0, 0.0, 0.0]]


    })
    return cfg