
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.joblib"

FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "BILL_AMT1",
    "PAY_AMT1",
]

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Load model once per Python worker, not once per prediction.
model = None
model_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model_error = "File model belum tersedia."
except Exception as exc:
    model_error = f"Model gagal dimuat: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        if model is None:
            error = (
                "Model belum dapat dimuat. Pastikan "
                "models/credit_risk_model.joblib tersedia "
                "dan versi dependensi sesuai."
            )
        else:
            try:
                row = {
                    "LIMIT_BAL": float(request.form["LIMIT_BAL"]),
                    "AGE": int(request.form["AGE"]),
                    "EDUCATION": int(request.form["EDUCATION"]),
                    "MARRIAGE": int(request.form["MARRIAGE"]),
                    "PAY_0": int(request.form["PAY_0"]),
                    "PAY_2": int(request.form["PAY_2"]),
                    "PAY_3": int(request.form["PAY_3"]),
                    "BILL_AMT1": float(request.form["BILL_AMT1"]),
                    "PAY_AMT1": float(request.form["PAY_AMT1"]),
                }

                if row["LIMIT_BAL"] <= 0:
                    raise ValueError("Limit kredit harus lebih dari 0.")

                if not 21 <= row["AGE"] <= 100:
                    raise ValueError("Usia harus berada pada rentang 21–100.")

                if row["EDUCATION"] not in (1, 2, 3, 4):
                    raise ValueError("Pilihan pendidikan tidak valid.")

                if row["MARRIAGE"] not in (1, 2, 3):
                    raise ValueError("Pilihan status pernikahan tidak valid.")

                for feature in ("PAY_0", "PAY_2", "PAY_3"):
                    if not -2 <= row[feature] <= 8:
                        raise ValueError(
                            f"{feature} harus berada pada rentang -2 hingga 8."
                        )

                input_df = pd.DataFrame([row], columns=FEATURES)

                prediction = int(model.predict(input_df)[0])
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                # Cari probabilitas kelas gagal bayar (kelas 1).
                default_index = classes.index(1)
                probability = float(probabilities[default_index])

                result = {
                    "prediction": prediction,
                    "probability": probability * 100,
                    "is_default": prediction == 1,
                }

            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("Prediction failed")
                error = (
                    "Prediksi gagal diproses. Periksa input dan "
                    "pastikan model sesuai dengan fitur aplikasi."
                )

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_data=form_data,
    )


@app.route("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
    }, 200 if model is not None else 503


if __name__ == "__main__":
    app.run(debug=True)
