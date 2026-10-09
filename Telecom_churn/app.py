from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(
    open("model/modelRF.pkl", "rb")
)

# Load fitted OneHotEncoder
OHE = pickle.load(
    open("model/OHE.pkl", "rb")
)

# Original categorical columns
categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # -------------------------------
    # 1. Get categorical values
    # -------------------------------
    input_data = {
        "gender": request.form["gender"],
        "Partner": request.form["Partner"],
        "Dependents": request.form["Dependents"],
        "PhoneService": request.form["PhoneService"],
        "MultipleLines": request.form["MultipleLines"],
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "OnlineBackup": request.form["OnlineBackup"],
        "DeviceProtection": request.form["DeviceProtection"],
        "TechSupport": request.form["TechSupport"],
        "StreamingTV": request.form["StreamingTV"],
        "StreamingMovies": request.form["StreamingMovies"],
        "Contract": request.form["Contract"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "PaymentMethod": request.form["PaymentMethod"]
    }

    categorical_df = pd.DataFrame([input_data])

    # -------------------------------
    # 2. One-Hot Encode
    # -------------------------------
    encoded_data = OHE.transform(categorical_df)

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=OHE.get_feature_names_out(categorical_columns)
    )

    # -------------------------------
    # 3. Get numerical features
    # -------------------------------
    numeric_df = pd.DataFrame([{
        "SeniorCitizen": float(request.form["SeniorCitizen"]),
        "tenure": float(request.form["tenure"]),
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"])
    }])

    # -------------------------------
    # 4. Combine encoded + numeric
    # -------------------------------
    final_data = pd.concat(
        [encoded_df, numeric_df],
        axis=1
    )

    # Make sure feature order is same as training
    final_data = final_data[
        list(OHE.get_feature_names_out(categorical_columns))
        + [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    ]

    # -------------------------------
    # 5. Prediction
    # -------------------------------
    prediction = model.predict(final_data)[0]

    Churn = {
        0: "No",
        1: "Yes"
    }

    result = Churn[prediction]

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)