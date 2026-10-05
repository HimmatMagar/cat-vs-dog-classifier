import torch
from pathlib import Path
from Classifier import logger
from Classifier.utils import save_file
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Classifier.entity import ModelBuildingConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score



class ModelEval:

      def __init__(self, config: ModelBuildingConfig):
            self.config = config

      
      def prepare_data(self):
            test_data_transform = transforms.Compose([
                  transforms.Resize((128, 128)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_dataset = datasets.ImageFolder(self.config.test_data_file, transform=test_data_transform)
            return DataLoader(test_dataset, batch_size=32, shuffle=True)

      
      def evaluate(self, model, dataloader, device):
            correct, total = 0, 0
            with torch.no_grad():
                  for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        preds = model(x).argmax(1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
            return correct, total


      def evaluate_model(self):
            model = torch.load(self.config.model, weights_only=False)
            model.eval();

            test_dataset = self.prepare_data()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            correct, total = self.evaluate(model, test_dataset, device)

            performance_report = {
                  "accuracy" : accuracy_score(total, correct),
                  "precision": precision_score(total, correct, average="macro", zero_division=0),
                  "recall"   : recall_score(total, correct, average="macro", zero_division=0)
            }
            save_file(Path(self.config.metrices), performance_report)
            logger.info(f"Model performance report saved successfully in {self.config.metrices}")
            return performance_report