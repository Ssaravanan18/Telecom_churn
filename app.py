from flask import Flask, request, render_template
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open("model/DT_model1.pkl","rb"))
Encodelabel=pickle.load(open("model/LabelEncoding.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/pred",methods=["POST"])
def pred():

    features=[
        float(request.form["Gender"]),
        float(request.form["Age"]),
        float(request.form["AnnualSalary"])
    ]

    features=np.array([features])

    # features_encode=Encodelabel.fit_transform(features)
    prediction=model.predict(features)[0]

    car_purchase={0:"Purchased",1:"No_Purchased"}

    result=car_purchase[prediction]

    return render_template("index.html",prediction=result)

if __name__=="__main__":
    app.run(debug=True)