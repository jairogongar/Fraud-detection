import joblib
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import numpy as np

app = FastAPI()
model = joblib.load('model-0.1.0.pkl')   

@app.get('/')
def home():
    return {"Fraud detection": "OK"}

def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    if prediction == 1:
        label = "Fraud"
    
    else:
        label = "Ok"
    return {
        'label': label,
        'prediction': int(prediction),
        'probability': probability
    }

class InputData(BaseModel):
    SK_ID_CURR: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_INCOME_TYPE: int
    HOUR_APPR_PROCESS_START: int
    ORGANIZATION_TYPE: int



@app.post("/predict")
async def predict(input:InputData):
           
    response = get_model_response(input)
    return response
        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)