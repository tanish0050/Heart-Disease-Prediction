from flask import Flask
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# ------------------ LOAD DATA ------------------
heart_data = pd.read_csv("heart.csv", sep=",")
heart_data.columns = heart_data.columns.str.strip()

# ------------------ SPLIT DATA ------------------
X = heart_data.drop(columns='target')   # ✅ fixed (no axis)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# ------------------ MODEL ------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# ------------------ ACCURACY ------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(train_pred, Y_train))
print("Test Accuracy:", accuracy_score(test_pred, Y_test))

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return "Heart Disease Prediction App is Running 🚀"

@app.route("/predict")
def predict():
    input_data = (60,1,0,125,258,0,0,141,1,2.8,1,1,3)
    input_data = np.asarray(input_data).reshape(1,-1)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "Person has Heart Disease ❌"
    else:
        return "Person is Healthy ✅"

# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
