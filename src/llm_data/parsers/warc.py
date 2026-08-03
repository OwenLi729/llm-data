from dataclasses import dataclass
from typing import Optional

import cysimdjson
import daft
from daft import DataFrame, col

from llm_data.utils import decode_html


@daft.cls
class ExtractWarcLanguage:
    def __init__(self):
        self.parser = cysimdjson.JSONParser()

    def __call__(self, headers: str) -> str:
        try:
            element = self.parser.parse_string(headers)
            return element.at_pointer("/WARC-Identified-Content-Language")
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

extract_warc_language = ExtractWarcLanguage()


def select_warc_responses(df: DataFrame, limit: Optional[int] = None) -> DataFrame:
    df = df.where(col("WARC-Type") == "response")
    if limit is not None:
        df = df.limit(limit)
    return df


@dataclass
class ExtractWarc:
    limit: Optional[int] = None

    name: str = "ExtractWarc"

    def __call__(self, df: DataFrame) -> DataFrame:
        df = select_warc_responses(df, self.limit)
        df = df.with_column("html", decode_html(col("warc_content")))
        df = df.drop_null(col("html"))
        df = df.with_column("url", col("WARC-Target-URI"))
        df = df.with_column("record_id", col("WARC-Record-ID"))
        df = df.with_column("crawl_date", col("WARC-Date"))
        df = df.with_column("language", extract_warc_language(col("warc_headers")))
        return df.select(
            "html",
            "url",
            "record_id",
            "crawl_date",
            "language",
            "source_path",
        )
