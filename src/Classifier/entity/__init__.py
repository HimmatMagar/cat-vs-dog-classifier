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
      test_data_file: Path
      model: str
      batch_size: int
      num_worker: int
      epochs: int
      lr_layer: float
      lr_fc: float
      weight_decay: float

@dataclass(frozen=True)
class ModelEvalConfig:
      root_dir: Path
      test_data_file: Path
      model: Path
      metrices: Path
      batch_size: int
      num_worker: int