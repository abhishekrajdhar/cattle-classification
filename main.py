from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import shutil
import os

# Load YOLOv11 classification model
model = YOLO("best.pt")

# Initialize FastAPI
app = FastAPI(title="YOLOv11 Classification API", description="Serve YOLOv11 classification via FastAPI", version="1.0")

@app.get("/")
def home():
    return {"message": "YOLOv11 Classification API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    results = model.predict(temp_file)

    # Parse results
    predictions = []
    for r in results:
        top1 = r.probs.top1  # best class index
        conf = r.probs.top1conf.item()
        predictions.append({
            "class": model.names[top1],
            "confidence": float(conf)
        })

    # Clean up temp file
    os.remove(temp_file)

    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
