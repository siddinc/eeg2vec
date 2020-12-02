from flask import Flask, request
from constants import (
  UPLOAD_FOLDER_PATH,
  LOAD_MODEL_PATH,
  LOAD_DB_PATH
)
from utilities import (
  preprocess_request,
  get_data,
  contrastive_loss,
  custom_acc,
  load_siamese_net,
  load_db,
)
import json
import numpy as np
import time


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER_PATH
loaded_model = None
subject_data = None


@app.route("/api/predict", methods=["POST"])
def upload_file():
  if request.method == "POST":
    subject_id = int(request.form['subject_id'])
    data_no = int(request.form['data_no'])
    start_time = time.time()
    query_subject = get_data(subject_no=subject_id, data_no=data_no)
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
    end_time = time.time()
    final_result = [ i for i in subject_results[:6] if i[1]>0.7 ]
    response = { "results": final_result, "inference_time": round(end_time-start_time, 2) }
    return json.dumps(str(response))


if __name__ == "__main__":
  loaded_model = load_siamese_net(LOAD_MODEL_PATH + "/resnet.h5")
  print("[INFO]: Model Loaded")

  subject_data = load_db(LOAD_DB_PATH + "/test_dataset.csv")
  print("[INFO]: DB Loaded")

  app.run()