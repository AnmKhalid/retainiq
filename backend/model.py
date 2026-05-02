"""
ChurnPredictor — XGBoost + SHAP model for telecom customer churn.

Trains on the Kaggle Telco Customer Churn dataset format.
Features match the standard IBM/Kaggle telecom churn CSV schema.
"""

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import shap


# ── Feature config ────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

FEATURE_DESCRIPTIONS = {
    "tenure": "Account tenure (months)",
    "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
    "Contract": "Contract type",
    "InternetService": "Internet service type",
    "OnlineSecurity": "Online security add-on",
    "TechSupport": "Tech support add-on",
    "PaymentMethod": "Payment method",
    "PaperlessBilling": "Paperless billing",
}


class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.explainer = None
        self.feature_names = []
        self.accuracy = 0.0

    # ── Data preparation ──────────────────────────────────────────────────────
    def _preprocess(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        df = df.copy()

        # Coerce TotalCharges to numeric (blanks → median)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median() if fit else 0, inplace=True)

        # Encode categoricals
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[col] = "Unknown"
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le:
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0

        for col in NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0

        self.feature_names = CATEGORICAL_COLS + NUMERIC_COLS
        return df[self.feature_names].values.astype(float)

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, csv_path: str = None):
        """
        Train on the Kaggle Telco churn dataset.
        If csv_path is None, generates a synthetic dataset that mirrors
        the real dataset's statistical properties so the app works out-of-the-box.
        """
        if csv_path:
            df = pd.read_csv(csv_path)
        else:
            df = self._generate_synthetic_data(n=5000)

        # Encode target
        df["Churn"] = (df["Churn"].astype(str).str.upper().isin(["YES", "1", "TRUE"])).astype(int)

        X = self._preprocess(df, fit=True)
        y = df["Churn"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = self.model.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, preds), 4)
        print(f"[RetainIQ] Model trained — accuracy: {self.accuracy:.2%}")
        print(classification_report(y_test, preds, target_names=["Stay", "Churn"]))

        # Build SHAP tree explainer
        self.explainer = shap.TreeExplainer(self.model)

    # ── Prediction + SHAP ─────────────────────────────────────────────────────
    def predict_with_shap(self, df: pd.DataFrame) -> dict:
        customer_ids = df.get("customerID", pd.Series(range(len(df)))).tolist()
        X = self._preprocess(df, fit=False)
        probs = self.model.predict_proba(X)[:, 1]
        shap_values = self.explainer.shap_values(X)

        customers = []
        for i, (cid, prob) in enumerate(zip(customer_ids, probs)):
            sv = shap_values[i]
            # Build per-feature SHAP dict
            shap_dict = {
                FEATURE_DESCRIPTIONS.get(f, f): round(float(sv[j]), 4)
                for j, f in enumerate(self.feature_names)
            }
            top_factor_idx = np.argmax(np.abs(sv))
            top_factor = FEATURE_DESCRIPTIONS.get(
                self.feature_names[top_factor_idx], self.feature_names[top_factor_idx]
            )
            customers.append({
                "customer_id": str(cid),
                "churn_probability": round(float(prob), 4),
                "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
                "shap_values": shap_dict,
                "top_shap_feature": top_factor,
            })

        # Sort by churn probability descending
        customers.sort(key=lambda x: x["churn_probability"], reverse=True)

        high = sum(1 for c in customers if c["risk_level"] == "HIGH")
        medium = sum(1 for c in customers if c["risk_level"] == "MEDIUM")
        low = sum(1 for c in customers if c["risk_level"] == "LOW")

        # Aggregate SHAP importance
        shap_importance = {}
        for f_idx, f_name in enumerate(self.feature_names):
            label = FEATURE_DESCRIPTIONS.get(f_name, f_name)
            shap_importance[label] = round(float(np.mean(np.abs(shap_values[:, f_idx]))), 4)

        shap_importance_sorted = dict(
            sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "total_customers": len(customers),
            "high_risk": high,
            "medium_risk": medium,
            "low_risk": low,
            "model_accuracy": self.accuracy,
            "customers": customers,
            "global_shap_importance": shap_importance_sorted,
        }

    # ── Demo data ──────────────────────────────────────────────────────────────
    def get_demo_data(self) -> pd.DataFrame:
        return self._generate_synthetic_data(n=200)

    # ── Synthetic data generator ───────────────────────────────────────────────
    @staticmethod
    def _generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
        """
        Generates synthetic telecom data matching the Kaggle Telco schema.
        Replace with real CSV path in production for best accuracy.
        """
        rng = np.random.default_rng(42)
        tenure = rng.integers(0, 72, size=n)
        monthly = rng.uniform(20, 120, size=n).round(2)
        total = (tenure * monthly * rng.uniform(0.9, 1.1, size=n)).round(2)

        # Churn probability: higher for short tenure, high monthly, month-to-month contract
        contract_type = rng.choice(["Month-to-month", "One year", "Two year"],
                                    p=[0.55, 0.25, 0.20], size=n)
        churn_logit = (
            -2.5
            + 0.03 * (monthly - 60)
            - 0.04 * tenure
            + 1.2 * (contract_type == "Month-to-month").astype(float)
            + rng.normal(0, 0.5, size=n)
        )
        churn_prob = 1 / (1 + np.exp(-churn_logit))
        churn = (rng.uniform(size=n) < churn_prob).astype(int)

        return pd.DataFrame({
            "customerID": [f"CUST-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.integers(0, 2, size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], p=[0.9, 0.1], size=n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], p=[0.34, 0.44, 0.22], size=n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": contract_type,
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": ["Yes" if c else "No" for c in churn],
        })

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[RetainIQ] Model saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        print(f"[RetainIQ] Model loaded ← {path}")
