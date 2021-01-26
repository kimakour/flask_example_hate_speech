from flask import Flask, request, url_for, redirect, render_template, jsonify
from joblib import load
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load("text_classification.joblib")
cols = ["text"]


@app.route("/")
def home():
    return render_template("home_twitter.html")


@app.route("/lol", methods=["POST"])
def lol():
    text_feature = [x for x in request.form.values()]
    prediction = model.predict(text_feature)
    prediction = int(prediction[0])
    return render_template(
        "home_twitter.html", pred="Hate speech =  {}".format(prediction)
    )


if __name__ == "__main__":
    app.run(debug=True)
