from tensorflow.keras.models import load_model, model_from_json
from flask import Flask, request
from constants import (
  UPLOAD_FOLDER_PATH,
  LOAD_MODEL_PATH, TREE_PATH,
  INDEX_DICT_PATH,
  LATENT_DIM
)
from utilities import (
  preprocess_request,
  load_model,
  get_data,
)
import os
import pandas as pd
import numpy as np


# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER_PATH


# @app.route("/", methods=["POST"])
# def upload_file():
#   if request.method == "POST":
#     file = request.files["eeg"]
#     path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(path)
#     eeg_csv = pd.read_csv(path)
#     data = np.array(eeg_csv["# data"])
#     preprocessed_data = preprocess_request(data)
#     query_embedding = encoder.predict(preprocessed_data)
#     query_embedding = np.reshape(query_embedding, (LATENT_DIM,))
#     nearest_embeddings = search_embedding(query_embedding, tree, index_dict, 5)

# if __name__ == "__main__":
#   app.run()

if __name__ == "__main__":
  loaded_model = load_model(LOAD_MODEL_PATH + "/resnet_weights.h5", LOAD_MODEL_PATH + "/resnet.json")
  print("model loaded")

  subject_data = []

  csv_data = pd.read_csv("../database/test_dataset.csv")
  for i in range(109):
    subject_data.append(np.array(csv_data[str(i)][:3000]))

  query_subject = get_data(subject_no=3, data_no=1)
  query_subject = np.reshape(query_subject, (query_subject.shape[0], 1))

  subject_results = []
  input_pair_group = np.empty((109, 2, 3000, 1), dtype=np.float64)
  count = 0
  for i, input_data in enumerate(subject_data):
    input_data = np.reshape(input_data, (input_data.shape[0], 1))
    input_pair = np.array([query_subject, input_data])
    input_pair = np.reshape(input_pair, (1, 2, 3000, 1))
    input_pair_group[i,:,:,:] = input_pair
    
  pred = loaded_model(input_pair_group)
  print(pred.shape)
  # preds = 1.0 - np.reshape(preds, (preds.shape[0],))
    # subject_results.append((preds, count))
    # count += 1
  # subject_results.sort()

  # for i in subject_results[:5]:
    # print(i)