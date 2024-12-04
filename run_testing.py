from tf_record_CNN_spherical_gradcheckpoint_valid_pad import tf_record_CNN_spherical
import glob
import numpy as np
import json
import sys
from types import ModuleType


def allvars_filtered(offset=0):
    # Saves all variables in their current state and returns them as dict
    frame = sys._getframe(1 + offset)
    d = frame.f_globals
    d.update(frame.f_locals)
    # Filters out all modules and functions so the output can be serialized to a
    # JSON
    filtered = {key: value for key, value in d.items() if not callable(value) and
                not isinstance(value, ModuleType) and '__loader__' not in key}
    return filtered


train_path_pattern = './input/signal*.tfrecord'
bkgd_train_path_pattern = './input/bkgd*.tfrecord'
arch_ID = 1
path_to_model_folders = './model_weights'
path_to_this_model = path_to_model_folders + '/net' + str(arch_ID)
config_array = np.load(path_to_this_model + '/config_array.npy')
files = (glob.glob(path_to_this_model + '/*'))
files_filtered = [f for f in files if ".json" not in f]
num_files = len(files_filtered)
regularizer = None

# Save all variables to a JSON file
filtered_vars = allvars_filtered()
with open(path_to_this_model + '/config.json', 'w') as fp:
    json.dump(filtered_vars, fp)

test = tf_record_CNN_spherical(False, False, False, False, False, False, False,
                               False, False, False, False, True,
                               True, False, True, 100000, 1, train_path_pattern,
                               bkgd_train_path_pattern, arch_ID, config_array, files, num_files, path_to_this_model, None,
                               40, 5)
