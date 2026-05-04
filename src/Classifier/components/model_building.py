from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Classifier.entity import ModelBuildingConfig



class DataLoader:

      def __init__(self, config: ModelBuildingConfig):
            self.config = config
      

      def TransformImage(self):
            train_data_transform = transforms.Compose([
                  transforms.Resize((128, 128)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.ImageFolder(self.config.train_data_file, transform=train_data_transform)
            return DataLoader(train_dataset, batch_size=32, shuffle=True)


