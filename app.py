from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # This fixes the CORS error and allows Telecom.html to talk to Python

# Load your actual model
# Ensure 'churn_decision_tree.joblib' is in the exact same folder as this app.py file
try:
    model = joblib.load('churn_decision_tree.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"\n=== Prediction Request ===")
        print(f"Received data: {data}")
        
        # Map frontend input to model features
        # The model was trained with: age, usage_gb, complaints, tenure_months, plan_type_enc
        # Frontend sends: monthlyBill, dataUsage, complaints, tenure, contractType
        
        tenure = data.get('tenure', 12)
        monthly_bill = data.get('monthlyBill', 599)
        usage_gb = data.get('dataUsage', 10)
        complaints = data.get('complaints', 0)
        contract_type = data.get('contractType', '1 Year')
        
        # Age estimation: New customers are younger, long-tenure customers are older
        # Young customers (low tenure) are higher churn risk
        estimated_age = min(70, max(18, 18 + (tenure / 60) * 52))
        
        # Plan type: Prepaid (0) has higher churn, Postpaid (1) has lower churn
        # Use contract type as indicator: Month-to-Month = Prepaid (0), others = Postpaid (1)
        if contract_type == 'Month-to-Month':
            plan_type_enc = 0  # Prepaid - higher churn risk
        else:
            plan_type_enc = 1  # Postpaid - lower churn risk
        
        # Prepare the data with EXACT column names the model expects
        input_data = pd.DataFrame([{
            'age': estimated_age,
            'usage_gb': usage_gb,
            'complaints': complaints,
            'tenure_months': tenure,
            'plan_type_enc': plan_type_enc
        }])
        
        print(f"Feature mapping:")
        print(f"  age: {estimated_age:.1f} (from tenure {tenure} months)")
        print(f"  usage_gb: {usage_gb}")
        print(f"  complaints: {complaints}")
        print(f"  tenure_months: {tenure}")
        print(f"  plan_type_enc: {plan_type_enc} ({'Prepaid' if plan_type_enc == 0 else 'Postpaid'})")
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        print(f"Model output:")
        print(f"  Prediction (0/1): {prediction[0]}")
        print(f"  Probability (churn): {probability:.4f} ({probability*100:.2f}%)")
        
        # Determine risk label
        # Adjust thresholds based on actual model behavior - raw probabilities are typically low
        if probability > 0.55:
            label = 'High Risk'
        elif probability > 0.30:
            label = 'Medium Risk'
        else:
            label = 'Low Risk'
        
        print(f"  Risk Label: {label}\n")
        
        return jsonify({
            'score': round(probability * 100),
            'label': label,
            'factors': [
                f"Complaints: {complaints} calls",
                f"Tenure: {tenure} months",
                f"Data Usage: {usage_gb} GB",
                f"Bill Amount: ₹{monthly_bill:.0f}",
                f"Plan Type: {contract_type}"
            ]
        })
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Running on port 5000 with debug mode enabled so you can see errors in the terminal
    app.run(port=5000, debug=True)