from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import daft
from daft import col
from daft.functions import length
from ftfy import fix_text
from lxml import etree
from magic_html import GeneralExtractor
from resiliparse.extract.html2text import extract_plain_text

_ARCHIVE_BASE_URL = (
    "https://archive.org/download/stackexchange_20260331/stackexchange_20260331"
)
_HTML_PREFIX = "<html><body>"
_HTML_SUFFIX = "</body></html>"

# site : sha1 of the release .7z archive
_SITES: dict[str, str] = {
    "quant.stackexchange.com": "6df03f8745978abb6d35ed8c6408fe2523c53a6b",
    "money.stackexchange.com": "7513703962372537ef59417e30510174d343d116",
    "economics.stackexchange.com": "ee27814be73ae0cbe5589426d1c05964aba5f3cb",
}

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

def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _fetch_posts_xml(site: str, work_dir: Path, curl: str, seven_zip: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    archive = work_dir / f"{site}.7z"
    partial = work_dir / f"{site}.7z.part"
    posts_xml = work_dir / "Posts.xml"
    expected_sha1 = _SITES[site]

    if not archive.is_file():
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--show-error",
                "--continue-at",
                "-",
                "--retry",
                "3",
                "--retry-delay",
                "5",
                "--output",
                str(partial),
                f"{_ARCHIVE_BASE_URL}/{site}.7z",
            ],
            check=True,
        )
        actual = _sha1(partial)
        if actual != expected_sha1:
            raise RuntimeError(f"{partial} sha1 {actual} != {expected_sha1}")
        partial.replace(archive)
    else:
        actual = _sha1(archive)
        if actual != expected_sha1:
            raise RuntimeError(f"{archive} sha1 {actual} != {expected_sha1}")

    if not posts_xml.is_file() or posts_xml.stat().st_size == 0:
        subprocess.run(
            [seven_zip, "e", "-y", "-aoa", f"-o{work_dir}", str(archive), "Posts.xml"],
            check=True,
        )
    if not posts_xml.is_file() or posts_xml.stat().st_size == 0:
        raise RuntimeError(f"missing Posts.xml for {site}")
    return posts_xml

def _optional_int(attributes, name: str) -> int | None:
    value = attributes.get(name)
    return None if value is None else int(value)


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
                    "parent_id": _optional_int(attrs, "ParentId"),
                    "score": _optional_int(attrs, "Score"),
                    "accepted_answer_id": _optional_int(attrs, "AcceptedAnswerId"),
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
            _to_text(daft.lit(_HTML_PREFIX) + col("body") + daft.lit(_HTML_SUFFIX)),
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
    curl: str,
    seven_zip: str,
) -> Path:
    site_work = work_dir / f"release=20260331" / f"site={site}"
    posts_xml = _fetch_posts_xml(site, site_work, curl, seven_zip)
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
        .with_column(
            "question",
            col("title") + daft.lit("\n\n") + col("text"),
        )
        .select(
            col("id").alias("question_id"),
            "question",
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
        .select("question", "answer")
    )
    pairs.write_parquet(
        str(Path(output_path).parent / "qa_pairs"),
        compression="zstd",
        write_mode="overwrite",
    )

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and process finance Stack Exchange dumps."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", choices=tuple(_SITES), default=None,)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=10_000)
    args = parser.parse_args(argv)

    selected = tuple(_SITES if args.sites is None else args.sites)
    if len(set(selected)) != len(selected):
        parser.exit(status=1, message="error: sites must not contain duplicates\n")

    work = args.work_dir.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    curl, seven_zip = shutil.which("curl"), shutil.which("7z")
    if not curl or not seven_zip:
        parser.exit(status=1, message="error: curl and 7z required on PATH\n")

    try:
        work.mkdir(parents=True, exist_ok=True)
        out_root.mkdir(parents=True, exist_ok=True)
        for site in selected:
            path = _process_site(
                site,
                work,
                out_root,
                args.batch_size,
                args.min_length,
                args.max_length,
                curl,
                seven_zip,
            )
            qa_pairs(path)
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        etree.XMLSyntaxError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(status=1, message=f"error: {error}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
