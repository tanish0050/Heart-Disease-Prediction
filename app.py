from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model
model = pickle.load(open('heartdisease.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['thalach'])
        ]

        final_features = [np.array(features)]
        prediction = model.predict(final_features)

        output = "Heart Disease Detected 💔" if prediction[0] == 1 else "No Heart Disease ✅"

        return render_template('index.html', prediction_text=output)

    except Exception as e:
        return f"Error: {e}"

# Run app (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)