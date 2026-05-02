from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import os
from model import ChurnPredictor

app = Flask(__name__)
CORS(app)

predictor = ChurnPredictor()

# ── Train on startup if model not saved ──────────────────────────────────────
MODEL_PATH = "churn_model.pkl"
if os.path.exists(MODEL_PATH):
    predictor.load(MODEL_PATH)
else:
    predictor.train("telco_churn.csv")
    predictor.save(MODEL_PATH)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_accuracy": predictor.accuracy})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts a CSV file upload with customer data.
    Returns churn probabilities + SHAP values per customer.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    try:
        df = pd.read_csv(file)
        results = predictor.predict_with_shap(df)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo", methods=["GET"])
def demo():
    """Returns predictions on built-in demo data for dashboard preview."""
    demo_df = predictor.get_demo_data()
    results = predictor.predict_with_shap(demo_df)
    return jsonify(results)


@app.route("/api/export-pdf", methods=["POST"])
def export_pdf():
    """Generates a PDF report from the prediction results."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    data = request.json
    customers = data.get("customers", [])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("RetainIQ — Churn Risk Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Total customers analysed: {len(customers)}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Customer ID", "Churn Risk %", "Risk Level", "Top Risk Factor"]]
    for c in customers[:50]:  # cap at 50 rows for PDF
        risk = c.get("churn_probability", 0) * 100
        level = "HIGH" if risk > 70 else "MEDIUM" if risk > 40 else "LOW"
        top_factor = c.get("top_shap_feature", "N/A")
        table_data.append([c.get("customer_id", "N/A"), f"{risk:.1f}%", level, top_factor])

    table = Table(table_data, colWidths=[120, 100, 80, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="churn_report.pdf",
                     mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
