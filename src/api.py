from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from src.inference import predict

app = FastAPI(
    title="Multimodal Product Classifier",
    version="1.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



@app.get("/")
def home():

    return {
        "message":
        "Multimodal Product Classifier API"
    }



@app.post("/predict")
async def classify_product(

    title: str = Form(...),

    image: UploadFile = File(...)
):

    temp_dir = "temp"

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    file_path = os.path.join(
        temp_dir,
        image.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            image.file,
            buffer
        )

    results = predict(
        file_path,
        title
    )

    os.remove(
        file_path
    )

    return {

        "predicted_category":
            results[0]["category"],

        "confidence":
            results[0]["confidence"],

        "top_predictions":
            results
    }