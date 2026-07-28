from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import daft
from daft import col
from daft.functions import length
from ftfy import fix_text
from lxml import etree
from magic_html import GeneralExtractor
from resiliparse.extract.html2text import extract_plain_text

_RAW_COLUMNS = (
    "id",
    "post_type_id",
    "parent_id",
    "score",
    "accepted_answer_id",
    "title",
    "body",
)

_POST_COLUMNS = (
    "id",
    "post_type_id",
    "parent_id",
    "score",
    "accepted_answer_id",
    "title",
    "text",
)

@daft.func
def _to_text(html: str) -> str:
    cleaned = GeneralExtractor().extract(html)["html"]
    return fix_text(extract_plain_text(cleaned) or "")

def _read_posts(posts_xml: Path, batch_size: int) -> Iterator[daft.DataFrame]:
    records: list[dict] = []
    for _, element in etree.iterparse(
        str(posts_xml),
        events=("end",),
        tag="row",
        load_dtd=False,
        no_network=True,
        resolve_entities=False,
        recover=False,
        huge_tree=True,
    ):
        attrs = element.attrib
        post_type_id = int(attrs["PostTypeId"])
        if post_type_id in (1, 2):
            records.append(
                {
                    "id": int(attrs["Id"]),
                    "post_type_id": post_type_id,
                    "parent_id": (
                        int(attrs["ParentId"]) if "ParentId" in attrs else None
                    ),
                    "score": int(attrs["Score"]) if "Score" in attrs else None,
                    "accepted_answer_id": (
                        int(attrs["AcceptedAnswerId"])
                        if "AcceptedAnswerId" in attrs
                        else None
                    ),
                    "title": attrs.get("Title"),
                    "body": attrs.get("Body"),
                }
            )
            if len(records) >= batch_size:
                yield daft.from_pydict(
                    {
                        column: [record[column] for record in records]
                        for column in _RAW_COLUMNS
                    }
                )
                records = []
        element.clear(keep_tail=True)
        while element.getprevious() is not None:
            del element.getparent()[0]

    if records:
        yield daft.from_pydict(
            {column: [record[column] for record in records] for column in _RAW_COLUMNS}
        )


def _process_batch(
    dataframe: daft.DataFrame,
    min_length: int,
    max_length: int,
) -> daft.DataFrame:
    return (
        dataframe.with_column(
            "text",
            _to_text(
                daft.lit("<html><body>") + col("body") + daft.lit("</body></html>")
            ),
        )
        .where(
            (length(col("text")) > min_length) & (length(col("text")) < max_length)
        )
        .select(*_POST_COLUMNS)
    )

def _process_site(
    site: str,
    work_dir: Path,
    output_dir: Path,
    batch_size: int,
    min_length: int,
    max_length: int,
    posts_xml: str,
) -> Path:
    site_work = work_dir / f"release=20260331" / f"site={site}"
    out = output_dir / "release=20260331" / f"site={site}" / "posts"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    batches = 0
    for dataframe in _read_posts(posts_xml, batch_size):
        batches += 1
        _process_batch(dataframe, min_length, max_length).write_parquet(
            out,
            compression="zstd",
            write_mode="append",
        )
    if batches == 0:
        raise RuntimeError(f"no posts for {site}")

    shutil.rmtree(site_work, ignore_errors=True)
    return out

def qa_pairs(output_path: str | Path, min_score: int = 5) -> None:
    parquet_paths = sorted(Path(output_path).rglob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No Parquet shards found in {output_path}")
    df = daft.read_parquet([str(path) for path in parquet_paths])

    questions = (
        df.where((col("post_type_id") == 1) & (col("score") >= min_score))
        .select(
            col("id").alias("question_id"),
            col("title").alias("question_title"),
            col("text").alias("question_text"),
            "accepted_answer_id",
        )
    )
    answers = df.where(col("post_type_id") == 2).select(
        col("id").alias("answer_id"),
        col("parent_id").alias("question_id"),
        col("text").alias("answer"),
        col("score").alias("answer_score"),
    )
    pairs = (
        questions.join(answers, on="question_id")
        .where(
            (col("answer_score") >= min_score)
            | (col("answer_id") == col("accepted_answer_id"))
        )
        .select(
            col("question_title").alias("title"),
            col("question_text").alias("question"),
            "answer",
        )
    )
    pairs.write_parquet(
        str(Path(output_path).parent / "qa_pairs"),
        compression="zstd",
        write_mode="overwrite",
    )
