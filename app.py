import joblib
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
import uvicorn
import pandas as pd
import numpy as np
from users import check_login
from auth import AuthHandler
from schemas import AuthDetails

app = FastAPI()
model = joblib.load('model-0.1.0.pkl')   
auth_handler = AuthHandler()
users = []

@app.post('/register', status_code=201)
def register(auth_details: AuthDetails):
    if any(x['username'] == auth_details.username for x in users):
        raise HTTPException(status_code=400, detail='Username is taken')
    hashed_password = auth_handler.get_password_hash(auth_details.password)
    users.append({
        'username': auth_details.username,
        'password': hashed_password    
    })
    return


@app.post('/login')
def login(auth_details: AuthDetails):
    user = None
    for x in users:
        if x['username'] == auth_details.username:
            user = x
            break
    
    if (user is None) or (not auth_handler.verify_password(auth_details.password, user['password'])):
        raise HTTPException(status_code=401, detail='Invalid username and/or password')
    token = auth_handler.encode_token(user['username'])
    return { 'token': token }

@app.get('/')
async def protected(username=Depends(auth_handler.auth_wrapper)):
    return {'name':username}

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
