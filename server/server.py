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
  get_data,
  contrastive_loss,
  custom_acc,
)
import os
import pandas as pd
import numpy as np


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER_PATH

loaded_model = load_model(LOAD_MODEL_PATH + "/resnet.h5", custom_objects={"contrastive_loss": contrastive_loss, "custom_acc": custom_acc})
csv_data = pd.read_csv("../database/test_dataset.csv")

subject_data = []
for i in range(109):
  subject_data.append(np.array(csv_data[str(i)][:3000]))


@app.route("/", methods=["POST"])
def upload_file():
  if request.method == "POST":
    
    preprocessed_data = preprocess_request(data)

    query_subject = get_data(subject_no=subject_id, data_no=4)
    query_subject = np.reshape(query_subject, (query_subject.shape[0], 1))
    input_pair_group = np.zeros((109, 2, 3000, 1), dtype=np.float64)
    count = 0

    for i, input_data in enumerate(subject_data):
      input_data = np.reshape(input_data, (input_data.shape[0], 1))
      input_pair_group[i,0,:,:] = query_subject
      input_pair_group[i,1,:,:] = input_data
      
    pred = loaded_model([input_pair_group[:,0], input_pair_group[:,1]])
    pred = 1.0 - np.reshape(pred, (pred.shape[0],))

    subject_results = []

    for i, j in enumerate(pred):
      subject_results.append((i, j))

    subject_results.sort(key=lambda e: e[1], reverse=True)



if __name__ == "__main__":
  app.run()