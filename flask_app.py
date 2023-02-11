import joblib
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import numpy as np
from flask import Flask, request
import jsonify

app = Flask(__name__)
model = joblib.load('model-0.1.0.pkl')   



    




@app.route("/predict", methods=['POST'])
def predict():
          
    data=request.get_json()
    
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    result = {
        'prediction': int(prediction.tolist()),
        'probability': probability.tolist()
    }
    return jsonify(result)
    
if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port=9696)
    