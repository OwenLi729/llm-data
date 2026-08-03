from typing import Optional

import daft
from daft import CheckpointConfig, CheckpointStore, KeyFilteringSettings

from llm_data.utils import checkpoint_uri


class DataLoader:
    def __init__(
        self,
        loader_type: str = "parquet",
        checkpoint_path: Optional[str] = None,
        file_path_column_name: Optional[str] = "source_path",
        checkpoint_on: Optional[str] = "source_path",
        num_workers: Optional[int] = None,
        cpus_per_worker: Optional[float] = None,
        config: Optional[CheckpointConfig] = None,
    ):
        if config is not None and checkpoint_path is not None:
            raise ValueError("Pass either `config` or `checkpoint_path`, not both")

        if checkpoint_path is not None and checkpoint_on is None:
            raise ValueError("`checkpoint_on` is required when `checkpoint_path` is set")
 
        self.loader_type = loader_type
        self.file_path_column_name = file_path_column_name
        self.checkpoint_path = checkpoint_uri(checkpoint_path) if checkpoint_path else None
 
        if self.checkpoint_path:
            config = CheckpointConfig(
                store=CheckpointStore(self.checkpoint_path),
                on=checkpoint_on,
                settings=KeyFilteringSettings(
                    num_workers=num_workers,
                    cpus_per_worker=cpus_per_worker,
                ),
            )
        self.config = config

    def read_data(self, input_path: str):
        if self.loader_type == "parquet":
            df = daft.read_parquet(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "json":
            df = daft.read_json(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "csv":
            df = daft.read_csv(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "huggingface":
            df = daft.read_huggingface(input_path)
        elif self.loader_type == "lance":
            df = daft.read_lance(input_path, checkpoint=self.config)
        elif self.loader_type == "warc":
            df = daft.read_warc(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        else:
            raise ValueError("DataLoader not supported.")
        return df
