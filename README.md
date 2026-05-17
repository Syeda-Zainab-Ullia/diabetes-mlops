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
| Logistic Regression | 0.924092 | 0.913285 | 0.924092 | 0.916827 |
| SVM | 0.834983 | 0.697197 | 0.834983 | 0.759895 |
| Decision Tree | 0.966997 | 0.957361 | 0.966997 | 0.962106 |
| Random Forest | 0.976898 | 0.964080 | 0.976898 | 0.970323 |
| KNN | 0.887789 | 0.888420 | 0.887789 | 0.887513 |

## Screenshots
Stored inside screenshots folder.
