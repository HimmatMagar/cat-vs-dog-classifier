from Classifier import logger
from Classifier.config import ConfigurationManager
from Classifier.components.model_eval import ModelEval


STAGE_NAME = "Model Eval Stage"

class ModelEvalPipeline:

      def __init__(self):
            pass

      
      def main(self):
            config = ConfigurationManager()
            eval_config = config.get_model_eval_config()

            model_eval = ModelEval(eval_config)
            model_eval.evaluate_model()

if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
            obj = ModelEvalPipeline()
            obj.main()
            logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e 