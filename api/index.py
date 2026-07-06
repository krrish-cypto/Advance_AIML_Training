from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pathlib import Path

# Vercel looks for this 'app' variable to start the serverless function
app = Flask(__name__)
# Replace 'YOUR_VERCEL_URL' with your actual Vercel project URL (e.g., 'https://my-telecom-project.vercel.app')
# You can also use "*" to allow all, but it is less secure.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Dynamically find the model in the root directory
MODEL_PATH = Path(__file__).resolve().parents[1] / "churn_decision_tree.joblib"

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:
    MODEL = None
    MODEL_ERROR = str(exc)

@app.route('/', methods=['GET'])
def health_check():
    return "API is healthy!", 200
    if MODEL is None:
        return jsonify({"error": f"Model failed to load: {MODEL_ERROR}"}), 500

    try:
        # Get data from the frontend fetch request
        data = request.get_json() or {}
        tenure = data.get("tenure", 12)
        monthly_bill = data.get("monthlyBill", 599)
        usage_gb = data.get("dataUsage", 10)
        complaints = data.get("complaints", 0)
        contract_type = data.get("contractType", "1 Year")

        # Prepare the features exactly as your model expects
        estimated_age = min(70, max(18, 18 + (tenure / 60) * 52))
        plan_type_enc = 0 if contract_type == "Month-to-Month" else 1

        input_data = pd.DataFrame([{
            "age": estimated_age,
            "usage_gb": usage_gb,
            "complaints": complaints,
            "tenure_months": tenure,
            "plan_type_enc": plan_type_enc,
        }])

        # Run Prediction
        probability = MODEL.predict_proba(input_data)[0][1]

        if probability > 0.55:
            label = "High Risk"
        elif probability > 0.30:
            label = "Medium Risk"
        else:
            label = "Low Risk"

        return jsonify({
            "score": round(probability * 100),
            "label": label,
            "factors": [
                f"Complaints: {complaints} calls",
                f"Tenure: {tenure} months",
                f"Data Usage: {usage_gb} GB",
                f"Bill Amount: ₹{monthly_bill:.0f}",
                f"Plan Type: {contract_type}",
            ],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500