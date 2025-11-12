# Student Performance Prediction

Simple Streamlit app that implements Multiple Linear Regression (closed-form) and a batch Gradient Descent regressor on a student performance dataset.

## Files
- [app.py](app.py) — Streamlit app and model implementations (includes [`MLR`](app.py), [`GDRegressor`](app.py), [`stream_data`](app.py), scaler instance [`scaler`](app.py), and sklearn model `lr` in [app.py](app.py))
- [Multiple_Linear_Regression.ipynb](Multiple_Linear_Regression.ipynb) — Notebook with exploratory analysis and experiments
- [Student_Performance.csv](Student_Performance.csv) — Dataset (columns described below)

## Dataset
Columns in [Student_Performance.csv](Student_Performance.csv):
- Hours Studied
- Previous Scores
- Extracurricular Activities (Yes/No)
- Sleep Hours
- Sample Question Papers Practiced
- Performance Index (target)

## Requirements
Install dependencies:
```sh
pip install -r requirements.txt
```

## Run the Streamlit App
```sh
streamlit run app.py
```
Then open the provided local URL in your web browser.