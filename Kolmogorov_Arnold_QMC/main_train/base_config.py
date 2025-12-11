import ml_collections
from ml_collections import config_dict

def default() -> ml_collections.ConfigDict:

    cfg = ml_collections.ConfigDict({
        'batch_size': 128,
        'pos': [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
                [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]],
        'charges': [6.],
        'spins': [1, 1, 1, -1, -1, -1],
        'atoms': [[0.0, 0.0, 0.0]],
        'layer_dims': [4, 4, 4, 6],
        'g': [3, 3, 3,],
        'k': [3, 3, 3,],
        'grid_range':[[-10, 10], [-10, 10], [-10, 10]],
        'iterations': 1000,
        'preiterations': 1000,
        'chebyshev': True,
        'spline': False,
        'envelope_chebyshev': False,
        'envelope_spline': False,
        'envelope_simple': True,
        'add_residual' : False,
        'add_bias': True,
        'external_weights': True,
        'system':{
            'molecule': config_dict.placeholder(list),
            'electrons': (3, 3),
            'nelectrons': 6,

        },
        'envelope':{
            'g_envelope': 10,
            'k_envelope': 3,
            'grid_range_envelope': [-10, 10],
        }
    })
    return cfg