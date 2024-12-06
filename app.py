from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for API requests

# Load the saved model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route("/", methods=["GET"])
def home():
    # Render the HTML file for the UI
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Validate input
        required_fields = [
            "pregnancies", "glucose", "blood_pressure",
            "skin_thickness", "insulin", "bmi",
            "diabetes_pedigree", "age"
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing fields in input"}), 400

        # Convert input data to numpy array and scale
        input_data = np.array([[data["pregnancies"], data["glucose"], data["blood_pressure"],
                                data["skin_thickness"], data["insulin"], data["bmi"],
                                data["diabetes_pedigree"], data["age"]]])
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
