from Classifier import logger
from Classifier.constants import *
from Classifier.utils import *
from Classifier.entity import *


class ConfigurationManager:

      def __init__(self, config = config, params = params):
            self.config = read_yaml(config)
            self.params = read_yaml(params)
            
            create_dir([config.artifact_root])