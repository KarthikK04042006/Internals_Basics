from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    order_weight_kg: float = Field(..., ge=0.5, le=15)
    distance_km: float = Field(..., ge=0.5, le=10)
    is_peak_hour: int = Field(..., ge=0, le=1)
    items_count: int = Field(..., ge=1, le=20)

@app.get("/heartbeat")
def heartbeat():
    return {"status": "healthy", "model_loaded": True}

@app.post("/infer")
def infer(data: InputData):
    input_data = [[
        data.order_weight_kg,
        data.distance_km,
        data.is_peak_hour,
        data.items_count
    ]]
    prediction = model.predict(input_data)[0]
    return {"prediction": float(prediction)}