from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
import pickle
import random
import pandas as pd
import os


def preprocess_request(vector):
  normalized_vector = (vector - vector.min()) / (vector.max() - vector.min())
  reshaped_vector = np.reshape(normalized_vector, (1, vector.shape[0], 1))
  return reshaped_vector


# def load_model(weights_path, model_json_path):
#   with open(model_json_path, 'r') as json_file:
#     loaded_model_json = json_file.read()

#   loaded_model = model_from_json(loaded_model_json)
#   loaded_model.load_weights(weights_path)
#   return loaded_model


def get_data(subject_no=0, data_no=1):
  start = data_no*3000
  csv_data = pd.read_csv("../database/test_dataset.csv")
  subject_query = np.array(csv_data[str(subject_no)][start:start+3000])
  return subject_query


def contrastive_loss(y_true, y_pred):
  margin = 1.0
  return K.mean(y_true * K.square(y_pred) + (1.0 - y_true) * K.square(K.maximum(margin - y_pred, 0.0)))


def custom_acc(y_true, y_pred):
  return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def load_siamese_net(model_path):
  loaded_model = load_model(
    model_path,
    custom_objects={
      "contrastive_loss": contrastive_loss,
      "custom_acc": custom_acc
    }
  )
  return loaded_model


def load_db(db_path):
  csv_data = pd.read_csv(db_path)
  subject_data = []

  for i in range(109):
    subject_data.append(np.array(csv_data[str(i)][:3000]))

  return subject_data

