from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
import re
import base64
import pickle
from dense import Dense
from activations import Leaky_Relu, Softmax
from network import predict
from scipy.ndimage import center_of_mass, shift

app = Flask(__name__)

# Define your model structure here
network = [
    Dense(784, 128),
    Leaky_Relu(),
    Dense(128, 64),
    Leaky_Relu(),
    Dense(64, 10),
    Softmax()
]

# Load trained weights
def load_model(filename, network_structure):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    data_idx = 0
    for layer in network_structure:
        if isinstance(layer, Dense):
            layer.load(data[data_idx])
            data_idx += 1
    return network_structure

network = load_model("trained_model.pkl", network)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_digit():
    data_url = request.json["image"]
    content = re.sub('^data:image/.+;base64,', '', data_url)
    image_bytes = base64.b64decode(content)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    
    img_array = np.array(image).astype('float32') / 255.0
    cy, cx = center_of_mass(img_array)
    shift_y, shift_x = np.array(img_array.shape) // 2 - [cy, cx]
    img_array = shift(img_array, shift=(shift_y, shift_x), mode='constant')
    img_array = img_array.reshape(784, 1)

    result = predict(network, img_array)
    pred = int(np.argmax(result))
    return jsonify({"prediction": pred})

if __name__ == '__main__':
    app.run(debug=True)
