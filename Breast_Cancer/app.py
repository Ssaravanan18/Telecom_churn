from flask import Flask, request, render_template   # request--> Used to get data sent by the browser. # render_template--> Used to display an HTML page.
import pickle  # pickle is used to load your already-trained machine learning model
import numpy as np

app=Flask(__name__)   # Used to create your Flask application.  --> Flask creates your web application.

model=pickle.load(open("model/Ab_model.pkl","rb"))

feature_names = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",

    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",

    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    features=[]

    for name in feature_names:
        value=float(request.form[name])
        features.append(value)

    features=np.array([features])  # Machine-learning models expect the input in a particular 2D format: (number of samples, number of features).
    prediction=model.predict(features)[0]

    breast_cancer={0:"Cancer",1:"No Cancer"}

    result=breast_cancer[prediction]
    return render_template("index.html",prediction=result)

if __name__ == "__main__":
    app.run(debug=True)



# GET → usually used to request/view something.
# POST → usually used to send/submit data.
# What if you don't write methods=["POST"]?
# If you write: @app.route("/predict")
# Flask allows GET by default.
# But your HTML form is sending POST:
# <form action="/predict" method="POST">
# So Flask will reject it with:
# 405 Method Not Allowed