from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from predict import predict_data

app = FastAPI(title="Iris Classifier API", version="1.0")

class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, lt=10, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width in cm")

class IrisResponse(BaseModel):
    response: int
    species: str
    confidence: float

def to_features(item: IrisData):
    # Order must match sklearn's iris dataset
    return [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris: IrisData):
    try:
        return predict_data([to_features(iris)])[0]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/batch", response_model=list[IrisResponse])
async def predict_batch(samples: list[IrisData]):
    if not samples:
        raise HTTPException(status_code=400, detail="Send at least one sample.")
    if len(samples) > 100:
        raise HTTPException(status_code=400, detail="Max 100 samples per request.")
    try:
        return predict_data([to_features(s) for s in samples])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))