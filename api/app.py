import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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


CLASS_NAMES = ["cat", "dog"]


@app.post("/predict")
async def predict_(file: UploadFile = File(...)):
    image_byte = await file.read()
    image = Image.open(io.BytesIO(image_byte)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output     = model(tensor)
        probs      = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "class"     : CLASS_NAMES[predicted.item()],
        "confidence": round(confidence.item(), 4)
    }
