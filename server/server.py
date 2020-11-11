from tensorflow.keras.models import load_model
from flask import Flask, request
from constants import (
  UPLOAD_FOLDER_PATH,
  LOAD_MODEL_PATH, TREE_PATH,
  INDEX_DICT_PATH,
  LATENT_DIM
)
from utilities import (
  preprocess_request,
  read_database,
  search_embedding
)
import os
import pandas as pd
import numpy as np


encoder = load_model(LOAD_MODEL_PATH + "encoder.h5")
tree, index_dict = read_database(TREE_PATH, INDEX_DICT_PATH)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER_PATH


@app.route("/", methods=["POST"])
def upload_file():
  if request.method == "POST":
    file = request.files["eeg"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)
    eeg_csv = pd.read_csv(path)
    data = np.array(eeg_csv["# data"])
    preprocessed_data = preprocess_request(data)
    query_embedding = encoder.predict(preprocessed_data)
    query_embedding = np.reshape(query_embedding, (LATENT_DIM,))
    nearest_embeddings = search_embedding(pred, tree, index_dict, 5)

if __name__ == "__main__":
  app.run()