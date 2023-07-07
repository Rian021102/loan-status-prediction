import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json
import joblib
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Create the app object
app = FastAPI()

# Load the saved model
model = joblib.load("/Users/rianrachmanto/pypro/project/loan-status-prediction/model/model.pkl")

# Create a class that describes the input
class LoanStatus(BaseModel):
    emp_length_int: float
    home_ownership_cat: int
    annual_inc: int
    loan_amount: int
    term_cat: int
    application_type_cat: int
    purpose_cat: int
    interest_payment_cat: int
    interest_rate: float
    grade_cat: int
    dti: float
    total_pyment: float
    recoveries: float
    region: int
    

# Define the routes
@app.get('/')
def index():
    return {'message': 'Hello, Everyone!'}

@app.get('/Welcome/{name}')
def get_name(name: str):
    return {'Welcome to my ML model': name}

@app.post('/predict')
def predict_loan(data: LoanStatus):
    # Convert the input data to a dictionary and extract the values
    data_dict = data.dict()
    data_values = list(data_dict.values())

    # Convert the values to a 2D array
    input_data = np.array([data_values])

    # Make prediction
    prediction = model.predict(input_data)

    # Interpret the prediction
    if prediction == 0:
        loan_condition_cat = "good-loan"
    else:
        loan_condition_cat = "bad-loan"

    return {"loan_status": loan_condition_cat}

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8080)
