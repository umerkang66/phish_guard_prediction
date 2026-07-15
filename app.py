from dotenv import load_dotenv
import os, sys

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

import pymongo
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from uvicorn import run
from fastapi.responses import Response, HTMLResponse
import pandas as pd

from src.exception import CustomException
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

from src.constants.training_pipeline_constants import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)
from inputs import PhishingPredictInput

client = pymongo.MongoClient(DATABASE_URL)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["pages"])
async def index():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(
            content=f"<h3>Error loading index.html: {str(e)}</h3>", status_code=500
        )


@app.get("/train", tags=["training"])
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response(content="Training successful", status_code=200)
    except Exception as e:
        raise CustomException(e, sys)


@app.get("/predict", response_class=HTMLResponse, tags=["pages"])
async def predict_page():
    try:
        with open("templates/predict.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(
            content=f"<h3>Error loading predict.html: {str(e)}</h3>", status_code=500
        )


@app.post("/predict", tags=["prediction"])
async def predict_route(input_data: PhishingPredictInput):
    try:
        # Convert input Pydantic model to dataframe
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])

        # Instantiate PredictionPipeline
        pred_pipeline = PredictionPipeline()
        if not pred_pipeline.model_files_exist():
            return Response(
                content="Model or Preprocessing objects do not exist. Please train the model first.",
                status_code=400,
            )

        # Run Prediction
        pred_val = pred_pipeline.predict(df)

        # interpretation: 0 represents Phishing (-1), and 1 represents Legitimate (1).
        result_label = "Legitimate" if pred_val == 1 else "Phishing"

        return {"prediction": pred_val, "label": result_label, "status": "success"}
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run(app, host="localhost", port=8000)
