from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request model
class InputData(BaseModel):
    x: int

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.post("/predict")
def predict(data: InputData):
    # Logic xử lý: x nhân 2
    result = data.x * 2
    return {"result": result}