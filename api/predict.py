import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "churn_decision_tree.joblib"

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:  # pragma: no cover - runtime safety
    MODEL = None
    MODEL_ERROR = str(exc)
else:
    MODEL_ERROR = None


def _json_response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(payload),
    }


def _read_payload(request):
    if isinstance(request, dict):
        body = request.get("body", {})
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}
        return body or {}

    if hasattr(request, "body"):
        body = request.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}
        return body or {}

    if hasattr(request, "get_json"):
        try:
            return request.get_json() or {}
        except Exception:
            return {}

    return {}


def handler(request):
    if getattr(request, "method", None) == "OPTIONS":
        return _json_response(200, {"ok": True})

    if MODEL is None:
        return _json_response(500, {"error": f"Model failed to load: {MODEL_ERROR}"})

    try:
        data = _read_payload(request)
        tenure = data.get("tenure", 12)
        monthly_bill = data.get("monthlyBill", 599)
        usage_gb = data.get("dataUsage", 10)
        complaints = data.get("complaints", 0)
        contract_type = data.get("contractType", "1 Year")

        estimated_age = min(70, max(18, 18 + (tenure / 60) * 52))

        if contract_type == "Month-to-Month":
            plan_type_enc = 0
        else:
            plan_type_enc = 1

        input_data = pd.DataFrame([{
            "age": estimated_age,
            "usage_gb": usage_gb,
            "complaints": complaints,
            "tenure_months": tenure,
            "plan_type_enc": plan_type_enc,
        }])

        probability = MODEL.predict_proba(input_data)[0][1]

        if probability > 0.55:
            label = "High Risk"
        elif probability > 0.30:
            label = "Medium Risk"
        else:
            label = "Low Risk"

        return _json_response(200, {
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
    except Exception as exc:  # pragma: no cover - runtime safety
        return _json_response(500, {"error": str(exc)})
