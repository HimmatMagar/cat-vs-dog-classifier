from Classifier import logger
from Classifier.config import ConfigurationManager
from Classifier.components.train import TrainModel


stage_name = "Model Training Stage"

class ModelTrainingPipeline:

      def __init__(self):
            pass
      
      def Train_Model(self):
            config = ConfigurationManager()
            train_model_config = config.get_model_building_config()

            trainModel = TrainModel(train_model_config)
            trainModel.trainModel()


if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {stage_name} started <<<<<<")
            obj = ModelTrainingPipeline()
            obj.Train_Model()
            logger.info(f">>>>>> {stage_name} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e