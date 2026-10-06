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

            configure_mlflow(experiment_name="Pytorch-ResNET")

            with mlflow.start_run(run_name="ResNet18") as run:
                  try:
                        mlflow.log_params({
                              "batch_size": train_model_config.batch_size,
                              "num_worker": train_model_config.num_worker,
                              "epochs": train_model_config.epochs,
                              "lr_layer": train_model_config.lr_layer,
                              "lr_fc": train_model_config.lr_fc,
                              "weight_decay": train_model_config.weight_decay
                        })
                        logger.info("Parameters logged successfully!!!")
                        trainModel = TrainModel(train_model_config)
                        model = trainModel.trainModel()

                        model.eval();
                        mlflow.pytorch.log_model(
                              pytorch_model=model,
                              artifact_path="model",
                              serialization_format="pickle",
                              registered_model_name="ResNet-18"
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