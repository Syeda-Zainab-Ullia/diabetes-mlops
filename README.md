# Diabetes Prediction MLOps Project

## Project Description
This project predicts whether a patient is diabetic or non-diabetic using machine learning and FastAPI deployment.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Matplotlib
- Seaborn

## Setup Instructions

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Example CURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"age":65,"urea":7.5,"cr":52,"hba1c":11.2,"chol":6.1,"tg":2.8,"hdl":0.9,"ldl":3.5,"vldl":1.2,"bmi":32.5,"gender":"M"}'
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|---------|----------|------------|---------|----|
| Logistic Regression | add_result | add_result | add_result | add_result |
| SVM | add_result | add_result | add_result | add_result |
| Decision Tree | add_result | add_result | add_result | add_result |
| Random Forest | add_result | add_result | add_result | add_result |
| KNN | add_result | add_result | add_result | add_result |

## Screenshots

Stored inside screenshots folder.
