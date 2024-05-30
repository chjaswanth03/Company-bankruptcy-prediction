from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the trained model
model = joblib.load(r"C:\Users\JASWANTH CH\OneDrive\Desktop\@SEM 6\Summer Intership\Projects 2024 (03-04-2024)\Company bankruptcy prediction\Machine Learning code\model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the data from the form
        data = [float(request.form.get(f)) for f in request.form]
        # Reshape the data to match the model's input shape
        data = np.array(data).reshape(1, -1)
        # Predict using the loaded model
        prediction = model.predict(data)
        result = 'Bankruptcy' if prediction[0] == 1 else 'No Bankruptcy'
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
