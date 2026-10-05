import os
import torch
import torch.nn as nn
import torch.optim as optim
from Classifier import logger
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Classifier.entity import ModelBuildingConfig

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TrainModel:
      def __init__(self, config: ModelBuildingConfig):
            self.config = config
            self.device = get_device()
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model = self.model.to(self.device)

      
      def prepare_data(self):
            train_tf = transforms.Compose([
                  transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            test_tf = transforms.Compose([
                  transforms.Resize(128),
                  transforms.CenterCrop(128),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            train_dataset = datasets.ImageFolder(self.config.train_data_file, transform=train_tf)
            test_dataset = datasets.ImageFolder(self.config.test_data_file, transform=test_tf)

            pin = self.device
            train_data_loader = DataLoader(
                  train_dataset, batch_size=self.config.batch_size,
                  shuffle=True, num_workers=self.config.num_worker,
                  pin_memory= pin
            )
            test_data_loader = DataLoader(
                  test_dataset, batch_size=self.config.batch_size,
                  shuffle=False, num_workers=self.config.num_worker,
                  pin_memory= pin
            )
            return train_data_loader, test_data_loader

      
      def _configure_fc_layer(self, num_classes: int) -> None:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes).to(self.device)
      
      
      def _freeze_params(self) -> None:
            for p in self.model.parameters():
                  p.requires_grad = False
            for p in self.model.layer4.parameters():
                  p.requires_grad = True
            for p in self.model.fc.parameters():
                  p.requires_grad = True

      
      def trainModel(self):
            train_data_loader, test_data_loader = self.prepare_data()
            self._configure_fc_layer(num_classes=2)
            self._freeze_params()
            logger.info("Total trainable parameter %s", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam([
                  {"params": self.model.layer4.parameters(), "lr": self.config.lr_layer},
                  {"params": self.model.fc.parameters(),     "lr": self.config.lr_fc}
            ], weight_decay=self.config.weight_decay)

            for epoch in range(self.config.epochs):
                  ## training
                  self.model.train()
                  running_loss = 0
                  for x, y in train_data_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        optimizer.zero_grad()
                        loss = criterion(self.model(x), y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * x.size(0)
                  epochs_loss = running_loss / len(train_data_loader.dataset)

                  ## Validation
                  self.model.eval()
                  correct, total = 0, 0
                  with torch.no_grad():
                        for x, y in test_data_loader:
                              x, y = x.to(self.device), y.to(self.device)
                              preds = self.model(x).argmax(1)
                              correct += (preds == y).sum().item()
                              total += y.size(0)
                  acc = correct / total
                  logger.info("Epoch %d: loss=%.4f val_acc=%.4f", epoch + 1, epochs_loss, acc)

                  path = os.path.join(self.config.root_dir, self.config.model)
                  torch.save(self.model.state_dict(), path)
                  logger.info("Checkpoint saved to %s", path)
                  return self.model