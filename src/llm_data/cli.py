import argparse
from pathlib import Path
from typing import Optional, Sequence

import daft

from llm_data.encoding import FixEncoding
from llm_data.engine import DataEngine
from llm_data.filters.filters import DropNullFilter, LengthFilter
from llm_data.loader import DataLoader
from llm_data.parsers.html import ParseHtml
from llm_data.parsers.warc import NormalizeWarc, select_warc_responses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local WARC file into cleaned Parquet text."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=100_000)
    parser.add_argument("--runner", choices=("native", "ray"), default="native")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--preview-count", type=int, default=3)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input_path.is_file():
        parser.error(f"Input WARC file does not exist: {args.input_path}")
    if args.output_path.exists():
        parser.error(f"Output path already exists: {args.output_path}")
    if not args.output_path.parent.exists():
        parser.error(
            f"Output parent directory does not exist: {args.output_path.parent}"
        )
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.min_length < 0:
        parser.error("--min-length cannot be negative")
    if args.max_length <= args.min_length:
        parser.error("--max-length must be greater than --min-length")
    if args.preview_count < 0:
        parser.error("--preview-count cannot be negative")


def configure_runner(runner: str) -> None:
    if runner == "ray":
        daft.set_runner_ray()
    else:
        daft.set_runner_native()


def build_pipeline(min_length: int, max_length: int) -> DataEngine:
    return (
        DataEngine()
        .add(NormalizeWarc())
        .add(ParseHtml(parser_type="trafilatura"))
        .add(DropNullFilter())
        .add(FixEncoding())
        .add(LengthFilter(min_len=min_length, max_len=max_length))
    )


def print_previews(df, preview_count: int) -> None:
    if preview_count == 0:
        return
    previews = df.select("url", "text").limit(preview_count).to_pydict()
    for index, (url, text) in enumerate(
        zip(previews["url"], previews["text"]), start=1
    ):
        excerpt = " ".join(text.split())[:200]
        print(f"[{index}] {url}\n{excerpt}")


def run(args: argparse.Namespace) -> None:
    configure_runner(args.runner)
    checkpoint_path = (
        str(args.checkpoint_path) if args.checkpoint_path is not None else None
    )
    loader = DataLoader(
        loader_type="warc",
        checkpoint_path=checkpoint_path,
        checkpoint_on="source_path" if checkpoint_path else None,
    )
    source_df = loader.read_data(str(args.input_path))
    candidate_df = select_warc_responses(source_df, args.limit)
    input_count = candidate_df.count_rows()

    output_df = build_pipeline(
        min_length=args.min_length,
        max_length=args.max_length,
    ).run(candidate_df)
    output_df = output_df.select(
        "text",
        "url",
        "record_id",
        "crawl_date",
        "language",
        "source_path",
    )
    kept_count = output_df.count_rows()

    print(f"Candidate response records: {input_count}")
    print(f"Kept records: {kept_count}")
    print(f"Rejected records: {input_count - kept_count}")
    print_previews(output_df, args.preview_count)
    output_df.write_parquet(str(args.output_path))
    print(f"Wrote Parquet dataset: {args.output_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    run(args)


if __name__ == "__main__":
    main()
