
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

import os

if os.path.exists("house_price_model.pkl"):
    model = joblib.load("house_price_model.pkl")
else:
    from train_model import train_and_save_model
    model = train_and_save_model()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        features = {
            "MedInc": float(request.form["MedInc"]),
            "HouseAge": float(request.form["HouseAge"]),
            "AveRooms": float(request.form["AveRooms"]),
            "AveBedrms": float(request.form["AveBedrms"]),
            "Population": float(request.form["Population"]),
            "AveOccup": float(request.form["AveOccup"]),
            "Latitude": float(request.form["Latitude"]),
            "Longitude": float(request.form["Longitude"])
        }

        input_data = pd.DataFrame([features])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction * 100000, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
