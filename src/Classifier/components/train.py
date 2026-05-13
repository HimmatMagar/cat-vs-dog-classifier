import os
import torch
import torch.nn as nn
import torch.optim as optim
from Classifier import logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Classifier.entity import ModelBuildingConfig
from Classifier.components.model import CNeuralNetwork


class TrainModel:

      def __init__(self, config: ModelBuildingConfig):
            self.config = config
      

      def prepare_data(self):
            train_data_transform = transforms.Compose([
                  transforms.Resize((128, 128)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.ImageFolder(self.config.train_data_file, transform=train_data_transform)
            return DataLoader(train_dataset, batch_size=32, shuffle=True)
      

      def trainModel(self):
            model = CNeuralNetwork()
            train_data_loader = self.prepare_data()
            

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(5):

                  total_epochs_loss = 0
                  for image, labels in train_data_loader:
                        output = model(image)

                        loss = criterion(output, labels)

                        optimizer.zero_grad()

                        loss.backward()

                        optimizer.step()

                        total_epochs_loss += loss.item()
                  avg_loss = total_epochs_loss/len(train_data_loader)
                  print(f"Epochs: {epoch + 1}, loss: {avg_loss:.4f}")
            
            model_path = os.path.join(self.config.root_dir, self.config.model)
            with open(model_path, "wb") as f:
                  torch.save(model, f)
            logger.info(f"Model Saved successfully in {model_path}")
            
            


      


