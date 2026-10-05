import os
import torch
import torch.nn as nn
import torch.optim as optim
from Classifier import logger
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Classifier.entity import ModelBuildingConfig


class TrainModel:

      def __init__(self, config: ModelBuildingConfig):
            self.config = config
            self.model = models.resnet18(weight='DEFAULT')
      

      def prepare_data(self):
            data_transform = transforms.Compose([
                  transforms.Resize((128, 128)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.ImageFolder(self.config.train_data_file, transform=data_transform)
            test_dataset = datasets.ImageFolder(self.config.test_data_file, transform=data_transform)
            train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            return train_data_loader, test_data_loader
      

      def trainModel(self):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            for p in self.model.parameters():
                  p.requires_grad = False
            for p in self.model.layer4.parameters():
                  p.requires_grad = True

            model = self.model.to(device)
            train_data_loader, test_data_loader = self.prepare_data()    

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam([
                  {"params": model.layer4.parameters(), "lr": 1e-5},
                  {"params": model.fc.parameters(),     "lr": 1e-4}
            ], weight_decay=1e-2)

            for epoch in range(5):
                  ## training
                  model.train()
                  for x, y in train_data_loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(x), y)
                        loss.backward()
                        optimizer.step()

                  ## Validation
                  model.eval()
                  correct, total = 0, 0
                  with torch.no_grad():
                        for x, y in test_data_loader:
                              x, y = x.to(device), y.to(device)
                              preds = model(x).argmax(1)
                              correct += (preds == y).sum().item()
                              total += y.size(0)
                  print(f"Epoch {epoch+1}: val acc = {correct/total:.4f}")
            
            model_path = os.path.join(self.config.root_dir, self.config.model)
            with open(model_path, "wb") as f:
                  torch.save(model, f)
            logger.info(f"Model Saved successfully in {model_path}")
            return model