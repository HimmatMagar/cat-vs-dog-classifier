import io
import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


## Load the model
model = torch.load("artifact/model/model.pth", weights_only=False)
model.eval()

transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
