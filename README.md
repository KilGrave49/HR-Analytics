📊 HR Analytics Pro: Strategic Attrition Dashboard

🚀 Overview

HR Analytics Pro is a comprehensive Streamlit application designed to help HR Executives and Analysts visualize workforce data, identify root causes of attrition, and predict employee flight risk using Machine Learning.

This project was built to demonstrate:

Descriptive Analytics: High-level dashboards for executive reporting.

Diagnostic Analytics: Deep-dive tools to understand why employees leave (Salary, Stress, Workload).

Predictive Analytics: AI models (Random Forest, SVM, Logistic Regression, etc.) to forecast future attrition.

🌟 Key Features

1. 📊 Executive Dashboard

Sunburst Charts: Visualize attrition hierarchy (Dept → Role → Status).

3D Talent View: Analyze Satisfaction vs. Performance vs. Salary.

Real-time KPIs: Track Attrition Rate, Avg Tenure, and Satisfaction.

2. 🔎 Deep Dive Analytics

Salary Analysis: Violin plots to detect pay disparities.

Stress & Burnout: KDE Density plots to correlate stress with turnover.

Turnover Reasons: Treemaps to categorize exit interview data.

3. 🤖 AI Prediction Engine

Algorithms: Support for Naive Bayes, Decision Tree, Random Forest, Logistic Regression, and SVM.

Performance Metrics: ROC Curves, Confusion Matrices, and Feature Importance.

Live Simulation: "What-if" analysis tool to predict risk for individual employees.

🛠️ Installation

Clone the repository:

git clone [https://github.com/your-username/hr-analytics-pro.git](https://github.com/KilGrave49/hr-analytics-pro.git)
cd hr-analytics-pro


Create a Virtual Environment:

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


Install Dependencies:

pip install -r requirements.txt


Run the App:

streamlit run app.py


📂 Project Structure

app.py: The main Streamlit application code.

dataset.csv: The HR dataset used for analysis.

requirements.txt: List of Python libraries required.

README.md: Project documentation.

🤝 Contributors
Akshay Kumar - Lead Developer & Analyst

Built as part of the Business Intelligence & Data Modelling (BIDM) Course.

