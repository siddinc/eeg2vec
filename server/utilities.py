from tensorflow.keras.models import model_from_json
import numpy as np
import pickle
import random
import pandas as pd
import os


def get_euclidean(x, y):
  euclidean_distance = np.sqrt(np.sum(np.power(x - y, 2)))
  return euclidean_distance


def normalize_embedding(x):
  return x / np.linalg.norm(x, axis=1, ord=1, keepdims=True)


def preprocess_request(vector):
  normalized_vector = (vector - vector.min()) / (vector.max() - vector.min())
  reshaped_vector = np.reshape(normalized_vector, (1, vector.shape[0], 1))
  return reshaped_vector


def np2csv(x, file_name):
    np.savetxt(file_name, x, delimiter=",", header="data")


def load_model(weights_path, model_json_path):
  with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(weights_path)
  return loaded_model


def get_data(subject_no=0, data_no=1):
  start = data_no*3000
  csv_data = pd.read_csv("../database/test_dataset.csv")
  subject_query = np.array(csv_data[str(subject_no)][start:start+3000])
  return subject_query