from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from Classifier.pipeline.prediction_pipeline import PredictionPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    app.state.pipeline = PredictionPipeline(class_names=["cat", "dog"])
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allowed origins
    allow_credentials=True,        # Allow cookies
    allow_methods=["*"],           # Allow all HTTP methods
    allow_headers=["*"],           # Allow all headers
)

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        return await run_in_threadpool(app.state.pipeline.predict, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))