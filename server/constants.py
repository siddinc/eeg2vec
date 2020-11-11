import os


TREE_PATH = os.path.abspath('../database/tree.cpickle')
INDEX_DICT_PATH = os.path.abspath('../database/index_dict.cpickle')

ENCODER_PATH = os.path.abspath('../models/encoder.h5')
LOAD_MODEL_PATH = os.path.abspath("../models")
UPLOAD_FOLDER_PATH = os.path.abspath("../database/eeg_recordings/")

LATENT_DIM = 512