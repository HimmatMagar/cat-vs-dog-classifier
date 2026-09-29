import mlflow
from Classifier import logger
from Classifier.config import ConfigurationManager
from Classifier.components.train import TrainModel
from Classifier.utils.mlflow_config import configure_mlflow, save_run_id


stage_name = "Model Training Stage"

class ModelTrainingPipeline:

      def __init__(self):
            pass
      
      def Train_Model(self):
            config = ConfigurationManager()
            train_model_config = config.get_model_building_config()

            configure_mlflow(experiment_name="pytorch-cnn")

            with mlflow.start_run(run_name="cnn-model") as run:
                  try:
                        trainModel = TrainModel(train_model_config)
                        model, input_example = trainModel.trainModel()

                        model.eval();
                        logged_model = mlflow.pytorch.log_model(
                              pytorch_model=model,
                              artifact_path="model",
                              registered_model_name="cnn-model",
                              input_example=input_example
                        )
                        logger.info("Model logged successfully")
                        save_run_id(run.info.run_id)
                  except Exception as e:
                        print(f"Training Failed {e}")
                        raise e


if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {stage_name} started <<<<<<")
            obj = ModelTrainingPipeline()
            obj.Train_Model()
            logger.info(f">>>>>> {stage_name} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e