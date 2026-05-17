from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import pandas as pd
import joblib

app = FastAPI()

# Load trained model and training columns
model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")


# --------------------------------
# Input Validation Schema
# --------------------------------
class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: str

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value):
        value = value.upper()

        if value not in ["M", "F"]:
            raise ValueError("Gender must be M or F")

        return value


# --------------------------------
# Health Check
# --------------------------------
@app.get("/")
def home():
    return {"status": "API is running"}


# --------------------------------
# Prediction Endpoint
# --------------------------------
@app.post("/predict")
def predict(data: PatientData):

    try:
        # Create input according to training dataset columns
        input_data = {
            "AGE": data.age,
            "Urea": data.urea,
            "Cr": data.cr,
            "HbA1c": data.hba1c,
            "Chol": data.chol,
            "TG": data.tg,
            "HDL": data.hdl,
            "LDL": data.ldl,
            "VLDL": data.vldl,
            "BMI": data.bmi,
            "Gender_M": 1 if data.gender == "M" else 0
        }

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # Match exact training columns
        df_input = df_input.reindex(
            columns=training_columns,
            fill_value=0
        )

        # Predict
        prediction = model.predict(df_input)[0]

        # Convert model output to readable result
        pred_text = str(prediction).upper()

        if pred_text in ["Y", "YES", "1"]:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"

        return {
            "prediction": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }
