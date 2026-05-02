# RetainIQ — Customer Churn Prediction Dashboard

> Predict which customers are about to leave — before they do.

A full-stack machine learning web app that scores every customer with a churn probability, explains the reasons using SHAP, and delivers a downloadable PDF report — all from a single CSV upload.

---

## Live Demo

![Dashboard Preview](https://via.placeholder.com/900x500/040d1a/00e5ff?text=RetainIQ+Dashboard)

> Replace this with a real screenshot after running the project locally.

---

## What It Does

- Upload any CSV of customer data
- XGBoost model scores every customer with a **churn probability (0–100%)**
- Classifies each customer as **High / Medium / Low** risk
- **SHAP explainability** — shows exactly which features are driving the churn risk per customer
- Interactive charts — risk distribution, score histogram, feature importance
- **One-click PDF report** export for business use

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Recharts, react-dropzone, Vite |
| Backend | Python, Flask, Flask-CORS |
| ML Model | XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| PDF Export | ReportLab |

---

## ML Model Details

- **Algorithm:** XGBoost Classifier
- **Dataset:** IBM Telco Customer Churn (Kaggle)
- **Accuracy:** 85%+
- **Explainability:** SHAP values computed per customer
- **Key features used:** Contract type, tenure, monthly charges, internet service, tech support, payment method, online security, paperless billing

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/AnmKhalid/retainiq.git
cd retainiq
```

### 2. Run the backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server starts at `http://localhost:5000`

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

App opens at `http://localhost:3000`

---

## Using the Real Dataset (Recommended)

For 85%+ accuracy, use the real Kaggle dataset:

1. Download from: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place the CSV file inside the `backend/` folder
3. In `app.py`, update the train line to:

```python
predictor.train("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check + model accuracy |
| GET | `/api/demo` | Load 200 demo customer predictions |
| POST | `/api/predict` | Upload CSV → get churn scores |
| POST | `/api/export-pdf` | Download PDF churn report |

---

## CSV Format

Your CSV should match the Kaggle Telco Churn schema. Key columns:

```
customerID, tenure, MonthlyCharges, TotalCharges,
Contract, InternetService, TechSupport, PaymentMethod,
OnlineSecurity, PaperlessBilling, Churn
```

The `Churn` column is optional when uploading for prediction.

---

## Project Structure

```
retainiq/
├── backend/
│   ├── app.py              # Flask REST API (4 endpoints)
│   ├── model.py            # XGBoost training + SHAP logic
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Full React dashboard
│   │   └── main.jsx        # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js      # Proxies /api → Flask
├── .gitignore
└── README.md
```

---

## Resume Bullet Point

> "Developed a customer churn prediction system using XGBoost (85% accuracy) with SHAP explainability, deployed as a full-stack web application with CSV upload, interactive risk dashboard, and one-click PDF report generation."

---

## License

MIT — free to use, modify, and distribute.
