import json
from dataclasses import dataclass
from typing import Optional

import daft
from daft import DataFrame, col

from llm_data.utils import decode_html


@daft.func
def decode_warc_content(content: bytes) -> str:
    return decode_html(content)


@daft.func
def extract_warc_language(headers: str) -> str:
    try:
        return json.loads(headers).get("WARC-Identified-Content-Language")
    except (AttributeError, TypeError, ValueError):
        return None


def select_warc_responses(df: DataFrame, limit: Optional[int] = None) -> DataFrame:
    df = df.where(col("WARC-Type") == "response")
    if limit is not None:
        df = df.limit(limit)
    return df


@dataclass
class NormalizeWarc:
    limit: Optional[int] = None

    name: str = "NormalizeWarc"

    def __call__(self, df: DataFrame) -> DataFrame:
        df = select_warc_responses(df, self.limit)
        df = df.with_column("html", decode_warc_content(col("warc_content")))
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
