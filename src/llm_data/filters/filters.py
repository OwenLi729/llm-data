from dataclasses import dataclass

from daft import DataFrame, col
from daft.functions import length


@dataclass
class DropNullFilter:
    input_column: str = "text"

    name: str = "DropNullFilter"

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.drop_null(col(self.input_column))


@dataclass
class LengthFilter:
    input_column: str = "text"
    max_len: int = 10000
    min_len: int = 0

    name: str = "LengthFilter"

    def __call__(self, df: DataFrame) -> DataFrame:
        text_len = length(col(self.input_column))
        return df.where((text_len < self.max_len) & (text_len > self.min_len))
