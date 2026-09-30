import io
import mlflow
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from Classifier.utils import logger


class PredictionPipeline:
    def __init__(self, model_uri: str = "models:/cnn-model@champion", class_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names  # e.g. ["cat", "dog"]

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # map_location is passed through to torch.load
        self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully on %s", self.device)

    def predict(self, image_bytes: bytes) -> dict:
        """Synchronous, so it can be run in a threadpool."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError("Uploaded file is not a valid image")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        idx = predicted.item()
        return {
            "class_id": idx,
            "class_name": self.class_names[idx] if self.class_names else None,
            "confidence": round(confidence.item(), 4),
        }