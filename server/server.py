import os
import pandas as pd
import numpy as np
# from keras.models import load_model
from flask import Flask, request

# location where files are uploaded
UPLOAD_FOLDER = "./eeg_recordings"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model remains loaded
# encoder = load_model()


# convert signal to a csv file to upload to server
# use it for client
def np2csv(x, fileName):
    np.savetxt(fileName, x, delimiter=",", header="data")


@app.route("/", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # get the csv file
        file = request.files["eeg"]
        # get the csv path
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        # save the image
        file.save(path)
        # read the csv file and extract signal
        eeg_csv = pd.read_csv(path)
        data = np.array(eeg_csv["# data"])
        x = np.reshape(data, (1, data.shape[0], 1))
        
        return str(x.shape)


if __name__ == "__main__":
    app.run()
