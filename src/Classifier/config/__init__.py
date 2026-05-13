from Classifier import logger
from Classifier.constants import *
from Classifier.utils import *
from Classifier.entity import *


class ConfigurationManager:

      def __init__(self, config = config):
            self.config = read_yaml(config)
            
            create_dir([self.config.artifact_root])

      
      def get_data_ingestion_config(self) -> DataIngestionConfig:
            config = self.config.data_ingestion
            create_dir([config.root_dir])

            return DataIngestionConfig(
                  root_dir=config.root_dir,
                  data_path=config.data_path,
                  zip_file=config.zip_file,
                  unzip_file=config.unzip_file
            )
      
      def get_model_building_config(self) -> ModelBuildingConfig:
            config = self.config.model_building
            create_dir([config.root_dir])

            return ModelBuildingConfig(
                  root_dir = config.root_dir,
                  train_data_file = config.train_data_file,
                  model = config.model
            )
      

      def get_model_eval_config(self) -> ModelEvalConfig:
            config = self.config.model_evaluation
            create_dir([config.root_dir])

            return ModelEvalConfig(
                  root_dir=config.root_dir,
                  test_data_file=config.test_data_file,
                  model=config.model,
                  metrices=config.metrices
            )