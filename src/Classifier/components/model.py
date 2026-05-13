import torch.nn as nn

class CNeuralNetwork(nn.Module):
      
      def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                  nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="same"),
                  nn.BatchNorm2d(32),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
                  nn.BatchNorm2d(128),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(128 * 16 * 16, 128),
                  nn.ReLU(),
                  nn.Dropout(0.4),
                  nn.Linear(128, 64),
                  nn.ReLU(),
                  nn.Dropout(0.4),
                  nn.Linear(64, 2)
            )
      

      def forward(self, x):
            X = self.features(x)
            output = self.fc(X)
            return output