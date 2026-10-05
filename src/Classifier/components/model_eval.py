import torch
from pathlib import Path
from Classifier import logger
from Classifier.utils import save_file
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Classifier.entity import ModelBuildingConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class ModelEval:
      def __init__(self, config: ModelBuildingConfig):
            self.config = config
            self.device = get_device()

      
      def _prepare_data(self):
            test_tf = transforms.Compose([
                  transforms.Resize(128),
                  transforms.CenterCrop(128),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            test_dataset = datasets.ImageFolder(self.config.test_data_file, transform=test_tf)
            return DataLoader(
                  test_dataset, batch_size=self.config.batch_size,
                  shuffle=False, num_workers=self.config.num_worker
            )

      
      def _evaluate(self, model, dataloader, device):
            all_preds, all_labels = [], []
            with torch.no_grad():
                  for x, y in dataloader:
                        x = x.to(device)
                        preds = model(x).argmax(1).cpu()
                        all_preds.extend(preds.tolist())
                        all_labels.extend(y.tolist())
            return all_labels, all_preds



      def _evaluate_model(self):
            model = torch.load(self.config.model, weights_only=False)
            model.to(self.device)
            model.eval();

            test_dataset = self._prepare_data()
            correct, total = self._evaluate(model, test_dataset, self.device)

            performance_report = {
                  "accuracy" : accuracy_score(total, correct),
                  "precision": precision_score(total, correct, average="macro", zero_division=0),
                  "recall"   : recall_score(total, correct, average="macro", zero_division=0)
            }
            save_file(Path(self.config.metrices), performance_report)
            logger.info(f"Model performance report saved successfully in {self.config.metrices}")
            return performance_report