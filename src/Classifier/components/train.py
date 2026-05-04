import torch.nn as nn
import torch.optim as optim
from Classifier import logger
from Classifier.components.model import CNeuralNetwork
from Classifier.components.model_building import DataLoader


class TrainModel:

      def __init__(self):
            model = CNeuralNetwork()
            dataloader = DataLoader()
            train_data_loader = dataloader.TransformImage()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(10):

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
            
            


      


