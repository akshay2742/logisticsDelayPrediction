from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Load the trained model
model = joblib.load('models/best_model.pkl')
data_prep = joblib.load('models/data_prep.pkl')

class ShipmentInput(BaseModel):
    origin: str
    destination: str
    vehicle_type: str
    distance: float
    weather: str
    traffic: str

@app.post("/predict")
async def predict_delay(shipment: ShipmentInput):
    try:
        # Transform input data
        input_data = {
            'Origin': data_prep.le_dict['Origin'].transform([shipment.origin])[0],
            'Destination': data_prep.le_dict['Destination'].transform([shipment.destination])[0],
            'Vehicle_Type': data_prep.le_dict['Vehicle_Type'].transform([shipment.vehicle_type])[0],
            'Distance': shipment.distance,
            'Weather_Conditions': data_prep.le_dict['Weather_Conditions'].transform([shipment.weather])[0],
            'Traffic_Conditions': data_prep.le_dict['Traffic_Conditions'].transform([shipment.traffic])[0]
        }
        
        # Make prediction
        features = np.array([list(input_data.values())])
        prediction = model.predict(features)[0]
        
        return {
            "delay_predicted": "Yes" if prediction == 1 else "No",
            "probability": float(model.predict_proba(features)[0][1])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Start the server when the script is called
    uvicorn.run(app, host="127.0.0.1", port=8000)