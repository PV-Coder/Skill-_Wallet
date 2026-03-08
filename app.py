from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("fraud_pipeline.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    deductable = float(request.form["deductable"])
    premium = float(request.form["premium"])
    umbrella = float(request.form["umbrella"])
    severity = float(request.form["severity"])
    claim = float(request.form["claim"])

    features = np.array([[age,deductable,premium,umbrella,severity,claim]])
    prob = model.predict_proba(features)[0][1]
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Fraudulent Claim"
        color = "red"
    else:
        result = "Genuine Claim"
        color = "green"

    return render_template(
        "index.html",
        prediction_text=result,
        color=color,
        age=age,
        deductable=deductable,
        premium=premium,
        umbrella=umbrella,
        severity=severity,
        claim=claim
    )

if __name__ == "__main__":
    app.run(debug=True)