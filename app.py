from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model + scaler
model = load_model("model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))

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
    app.run(debug=True)