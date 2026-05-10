from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 🔥 SAFE PATH FIX (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.keras")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = load_model(model_path)
scaler = pickle.load(open(scaler_path, "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    f1 = float(request.form['f1'])
    f2 = float(request.form['f2'])
    f3 = float(request.form['f3'])

    data = np.array([[f1, f2, f3]])
    data = scaler.transform(data)

    prediction = model.predict(data)
    result = np.argmax(prediction)

    actions = {
        0: "Turn Left",
        1: "Turn Right",
        2: "Move Forward",
        3: "Stop"
    }

    return render_template("index.html",
                           prediction_text=actions[result])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)