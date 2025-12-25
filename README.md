📡 Telecom Churn Prediction & Customer Retention Strategy
📋 Capstone Project Overview
This project focuses on designing and deploying a machine learning pipeline to predict customer churn for a telecom operator. The goal is to identify at-risk customers and generate actionable business insights to improve retention rates.

The project follows a full data science lifecycle: Data Engineering (ETL), Predictive Modeling, and Business Intelligence Visualization.

🛠️ Tech Stack
Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SQLAlchemy, PyArrow

Database: MySQL

Visualization: Microsoft Power BI

Environment: Jupyter Notebook / Anaconda

📂 Repository Structure
Plaintext

TELECOM_CHURN_CAPSTONE/
│
├── data_raw/                # Input datasets (Not always uploaded to git)
│   ├── customers.csv        # Customer demographics
│   ├── usage_data.csv       # Call and data usage logs
│   ├── complaints.csv       # Customer support tickets
│   └── billing.csv          # Contract and payment details
│
├── outputs/                 # Generated artifacts
│   ├── telecom_master.csv   # Cleaned and merged dataset
│   ├── model.pkl            # Trained ML model
│   ├── predictions.csv      # Dataset with churn probabilities
│   └── model_metrics.json   # Evaluation metrics
│
├── sql/
│   └── create_tables.sql    # Schema setup for MySQL
│
├── Telecom_Churn_Insights.pbix  # Power BI Dashboard file
├── capstone_churn.ipynb         # Main Jupyter Notebook
└── README.md                    # Project documentation
📊 Data Dictionary
The analysis is based on four primary data sources:

Customers: Profile, region, and plan type.

Usage Data: Data consumption (GB), calls made, and revenue generated.

Complaints: Complaint categories, timestamps, and resolution status.

Billing: Tenure, contract type (Month-to-month/Yearly), monthly charges, and churn status.

🚀 Project Pipeline
Phase 1: Data Engineering (ETL)
Ingestion: Loaded raw CSV files into Pandas.

Cleaning:

Handled missing values (Median imputation for numericals, "Resolved" for missing complaint status).

Standardized text fields (Region and Plan Type).

Converted Churn labels (Yes/No) to binary flags (1/0).

Integration: Aggregated complaint data to the customer level and merged all datasets.

Storage: Stored the final cleaned dataset (telecom_master.csv) and optionally uploaded it to a MySQL database (telecom_db).

Phase 2: Predictive Modeling
Features: Tenure, Monthly Charges, Data Usage, Calls Made, Complaint counts, Contract Type, etc.

Algorithms: Trained and compared Logistic Regression and Decision Tree Classifier.

Balancing: Used class_weight="balanced" to handle class imbalance inherent in churn data.

Evaluation: Optimized for F1-Score and Recall to ensure maximum capture of potential churners.

Output: The best-performing model is saved as model.pkl.

Phase 3: Visualization & Insights (Power BI)
A dashboard was created to visualize:

Churn Drivers: Analysis by Contract Type, Monthly Charges, and Tenure.

Regional Analysis: Churn rates across different geographic regions.

Complaint Trends: Correlation between open complaints and churn probability.

NLQ: Enabled "Ask a Question" for interactive data querying.

⚙️ Setup & Installation
1. Prerequisites
Ensure you have Python installed. It is recommended to use Anaconda.

Bash

pip install pandas numpy scikit-learn matplotlib seaborn jupyter pyarrow sqlalchemy pymysql
2. Database Setup (Optional)
If you wish to simulate the database integration step:

Install MySQL Workbench.

Run the script located in sql/create_tables.sql to create the schema.

3. Running the Analysis
Clone the repository.

Navigate to the project folder.

Open the Jupyter Notebook:

Bash

jupyter notebook capstone_churn.ipynb
Run all cells to perform ETL, train the model, and generate predictions.csv.

4. Viewing the Dashboard
Open Telecom_Churn_Insights.pbix using Microsoft Power BI Desktop.

Refresh the data source to point to the generated outputs/predictions.csv.

📈 Key Insights & Recommendations
Contract Sensitivity: Customers on "Month-to-Month" contracts show significantly higher churn rates compared to 1 or 2-year contracts.

Tenure Risk: The highest risk of churn occurs within the first few months of service.

Service Quality: Customers with unresolved "Open" complaints have a >80% probability of churning.

Strategy:

Incentivize long-term contracts for high-usage month-to-month users.

Prioritize ticket resolution for customers flagged with high churn probability.

📜 License
This project was developed as part of the Advanced AI/ML Training Capstone.
