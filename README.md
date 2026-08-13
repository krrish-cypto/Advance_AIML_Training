# 📡 Advance AI/ML Training & BSNL Telecom Churn Prediction

## 📋 Project Overview
This repository contains the complete capstone project and all corresponding exercises for the **Advance AI/ML Training**. 
The main capstone project focuses on designing and deploying a machine learning pipeline to predict customer churn for a telecom operator (BSNL context). The objective is to proactively identify at-risk customers and generate actionable business insights to improve retention rates.

The project features a full data science lifecycle, from **Data Engineering (ETL)**, **Predictive Modeling**, to **Frontend Dashboard Integration (React + Flask API)**.

---

## 🛠️ Tech Stack
- **Language**: Python 3.x, HTML/CSS/JavaScript
- **Machine Learning Libraries**: Pandas, NumPy, Scikit-learn, Joblib
- **Backend API**: Flask, Flask-CORS (Serverless Deployment ready for Vercel/Render)
- **Frontend**: React (via CDN), Tailwind CSS, Recharts
- **Environment**: Jupyter Notebook

---

## 📂 Repository Structure

### Capstone Artifacts
- `Telecom.html`: The main React-based frontend dashboard. Includes UI for viewing customer data, churn analytics, network QoS, and a live Churn Simulator.
- `api/index.py`: The Flask serverless API endpoint. It serves the ML model predictions to the frontend.
- `capstone_network.ipynb`: The primary Jupyter Notebook for Data Analysis, ETL, and Model Training.
- `churn_decision_tree.joblib`: The trained Decision Tree Classifier used by the backend API.
- `network_master.csv`: Cleaned dataset generated after ETL operations.
- `requirements.txt`: Python dependencies required for the backend API.
- `vercel.json`: Configuration for deploying the Flask API to Vercel.

### Training Modules (MOP Exercises)
The repository also includes multiple folders containing exercises covered during the Advance AI/ML Training:
- `Mop 2 - Exercise 1, 2, 3`
- `Mop 3 - Exercise 4, 5, 6, 8`
- `Mop 4 - Exercise 9, 10, 11`
- `MOP-20251225T...` & `Slides-20251225T...`: Additional training resources and slides.
- `Project PPT.pdf`: Final presentation slides for the capstone project.

---

## 🚀 Workflows & Architecture

### 1. Machine Learning Workflow (ETL & Modeling)
**File**: `capstone_network.ipynb`
1. **Data Ingestion**: Raw customer demographics, usage statistics, and complaints data are loaded into Pandas.
2. **Preprocessing**: 
   - Missing values are imputed.
   - Categorical fields (like Contract Type) are encoded.
   - Features relevant to churn (Tenure, Data Usage, Complaints, Monthly Bill) are engineered.
3. **Model Training**: A **Decision Tree Classifier** is trained to predict the likelihood of a customer churning based on historical data.
4. **Model Export**: The best-performing model is serialized and saved as `churn_decision_tree.joblib` for use in production.

### 2. Backend API Workflow (Flask Serverless)
**File**: `api/index.py`
1. **Model Loading**: The script dynamically resolves the path and loads the `churn_decision_tree.joblib` model.
2. **API Endpoint (`/` or `/api/predict`)**: Exposes a POST endpoint (or handles POST data).
3. **Prediction Generation**: 
   - Receives customer parameters (Tenure, Bill, Data Usage, Complaints, Contract Type) in JSON format from the frontend.
   - Preprocesses the inputs to match the model's expected format.
   - Calls `MODEL.predict_proba()` to compute the risk score (%).
   - Returns the score, risk label ("High Risk", "Medium Risk", "Low Risk"), and key factors contributing to the score as a JSON response.
4. **Deployment**: Configured for serverless deployment on **Vercel** (using `vercel.json`).

### 3. Frontend Dashboard Workflow (React + Tailwind)
**File**: `Telecom.html`
1. **Dashboard UI**: Built entirely in a single HTML file using React and Babel via CDN. 
2. **Components**:
   - **Dashboard**: High-level KPIs (Total Revenue, Active Customers, Churn Rate) and interactive charts (Recharts) for Revenue & Churn Trends.
   - **Customer Database**: A searchable table of customer records and their respective churn risk scores.
   - **Churn Simulator**: An interactive form where users can tweak variables (like Monthly Bill, Complaints) to see real-time churn predictions.
   - **Network QoS**: Visualizations correlating Network Signal Strength with customer complaints across regions.
3. **API Integration**: The "Run Prediction" button in the Churn Simulator makes an asynchronous `fetch` request to the deployed Flask backend (e.g., Render/Vercel URL) to retrieve live predictions using the trained model.

---

## ⚙️ Setup & Installation

### 1. Local Development (Backend)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/krrish-cypto/Advance_AIML_Training.git
   cd Advance_AIML_Training
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API**:
   Navigate to the `api` directory or run the Flask app locally:
   ```bash
   python api/index.py
   ```
   *(Note: You may need to adapt `index.py` with `app.run(debug=True)` to run it locally on `http://127.0.0.1:5000`)*.

### 2. Running the Frontend
1. Open the `Telecom.html` file directly in any modern web browser.
2. The UI will load immediately.
3. **Note**: For the "Churn Simulator" to work locally, ensure your backend is running and the `PREDICT_ENDPOINT` variable in `Telecom.html` is pointing to your local Flask server.

---

## 📈 Key Insights & Recommendations
- **Contract Sensitivity**: Customers on "Month-to-Month" contracts exhibit significantly higher churn rates.
- **Tenure Risk**: The highest risk of churn occurs within the first few months of service.
- **Service Quality**: Customers with a high number of recent complaints have a strongly elevated probability of churning.
- **Strategy**: Incentivize long-term contracts for high-usage month-to-month users and prioritize ticket resolution for high-risk customers.
