from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# 🔥 yaha change kiya hai
model = pickle.load(open('heartdisease.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        final_features = [np.array(features)]

        prediction = model.predict(final_features)

        output = "Heart Disease Detected 💔" if prediction[0] == 1 else "No Heart Disease ❤️"

    except Exception as e:
        return f"Error: {e}"

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)