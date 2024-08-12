from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import pandas as pd
import pathlib
class PredictionInput(BaseModel):
    length:float
    width:float
    Thickness:float
    Area:float
    Perimeter:float
    Roundness:float
    Solidity:float
    Compactness:float
    Eccentricity:float
    Extent:float
    Convex:float

curr_dir=pathlib.Path(__file__)
parent_dir=curr_dir.parent
app=FastAPI()
model_path='/Users/nirbhaysedha/Desktop/C/model.pkl'
model=load(model_path)

@app.get('/')
def home():
    return "hey we are on successfull!"

@app.post('/predict')
def predict(input_data: PredictionInput):
        features={
    "length":input_data.length,
    "width":input_data.width,
    "Thickness":input_data.Thickness,
    "Area":input_data.Area,
    "Perimeter":input_data.Perimeter,
    "Roundness":input_data.Roundness,
    "Solidity":input_data.Solidity,
    "Compactness":input_data.Compactness,
    "Eccentricity":input_data.Eccentricity,
    "Extent":input_data.Extent,
    "Convex":input_data.Convex,
        }
        feat=pd.DataFrame(features,index=[0])
        prediction = model.predict(feat)[0].item()
        return f"predictions : {prediction}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)




