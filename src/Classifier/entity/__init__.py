from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
      root_dir: Path
      data_path: str
      zip_file: Path
      unzip_file: Path


@dataclass(frozen=True)
class ModelBuildingConfig:
      root_dir: Path
      train_data_file: Path
      model: str