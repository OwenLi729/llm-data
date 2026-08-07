"""Cross-document 3-sentence span dedup (C4 rule), at document granularity.

C4 discards individual repeated 3-sentence spans; here we drop a whole
document once none of its spans are novel (i.e. every span already
occurred in an earlier document). This is simpler to express as a
distributed dataframe op and has the same boilerplate-removal effect
in practice.
"""
import re

import daft
from daft import Window, col

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@daft.func
def spans(text: str) -> list[str]:
    """Non-overlapping 3-sentence chunks of `text`."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    chunks = [" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3)]
    return chunks or [text]


def dedupe(df: daft.DataFrame, doc_id: str = "doc_id", text: str = "text") -> daft.DataFrame:
    """Drop documents whose 3-sentence spans have all appeared in an earlier document."""
    exploded = df.select(doc_id, text).with_column("span", spans(col(text))).explode("span")
    window = Window().partition_by("span").order_by(doc_id)
    ranked = exploded.select(doc_id, daft.functions.row_number().over(window).alias("rn"))
    novel_docs = ranked.where(col("rn") == 1).select(doc_id).distinct()
    return df.join(novel_docs, on=doc_id, how="inner")
