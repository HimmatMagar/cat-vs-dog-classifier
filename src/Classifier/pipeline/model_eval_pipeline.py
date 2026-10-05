import mlflow
from Classifier import logger
from dotenv import load_dotenv
from Classifier.config import ConfigurationManager
from Classifier.components.model_eval import ModelEval
from Classifier.utils.mlflow_config import configure_mlflow, load_run_id


STAGE_NAME = "Model Eval Stage"

class ModelEvalPipeline:

      def __init__(self):
            pass

      
      def main(self):
            config = ConfigurationManager()
            eval_config = config.get_model_eval_config()


            load_dotenv()
            run_id = load_run_id()

            with mlflow.start_run(run_id=run_id):
                  model_eval = ModelEval(eval_config)
                  metrices = model_eval._evaluate_model()

                  mlflow.log_metrics(metrices)
                  logger.info("metrices saved successfull")

if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
            obj = ModelEvalPipeline()
            obj.main()
            logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e 