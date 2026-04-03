import numpy as np
import pandas as pd
from flask import Flask
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# Load data
heart_data = pd.read_csv("heart.csv", sep=",")
heart_data.columns = heart_data.columns.str.strip()

# Split
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Test prediction
input_data = (60,1,0,125,258,0,0,141,1,2.8,1,1,3)
input_data = np.asarray(input_data).reshape(1,-1)

prediction = model.predict(input_data)
print("Prediction:", prediction)

# Run app
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)
if (prediction[0]==0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
