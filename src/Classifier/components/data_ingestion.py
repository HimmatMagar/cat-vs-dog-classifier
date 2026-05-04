import os
import gdown
import zipfile
import urllib.request as r
from Classifier import logger
from Classifier.entity import DataIngestionConfig


class DataIngestion:

      def __init__(self, config: DataIngestionConfig):
            self.config = config
      

      def download_zip_file(self):
            if not os.path.exists(self.config.zip_file):
                  file_id = "10GYovcJ9ga6cdQSKLXNFCOBhewZqWwYB"

                  gdown.download(
                        id=file_id,
                        output=self.config.zip_file,
                        quiet=False
                  )
                  logger.info("File downloaded successfully from google drive")
            
            else:
                  logger.info("File already exist")
      

      def extract_file(self):
            file = self.config.unzip_file
            os.makedirs(file, exist_ok=True)

            with zipfile.ZipFile(self.config.zip_file, "r") as f:
                  f.extractall(file)
