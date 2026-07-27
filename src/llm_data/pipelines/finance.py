from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import daft
from daft import col

if TYPE_CHECKING:
    from llm_data.engine import DataEngine

_LOGGER = logging.getLogger(__name__)

_RELEASE = "20260331"
_ARCHIVE_BASE_URL = (
    "https://archive.org/download/stackexchange_20260331/stackexchange_20260331"
)
_MANIFEST_VERSION = 1
_PARQUET_COMPRESSION = "zstd"
_ORIGINAL_BODY_COLUMN = "__original_body"
_HTML_DOCUMENT_PREFIX = "<html><body>"
_HTML_DOCUMENT_SUFFIX = "</body></html>"

_EXPECTED_SCHEMA = (
    ("id", "Int64"),
    ("post_type_id", "Int64"),
    ("accepted_answer_id", "Int64"),
    ("parent_id", "Int64"),
    ("score", "Int64"),
    ("view_count", "Int64"),
    ("owner_user_id", "Int64"),
    ("last_editor_user_id", "Int64"),
    ("answer_count", "Int64"),
    ("comment_count", "Int64"),
    ("favorite_count", "Int64"),
    ("site", "String"),
    ("creation_date", "String"),
    ("body", "String"),
    ("owner_display_name", "String"),
    ("last_editor_display_name", "String"),
    ("last_edit_date", "String"),
    ("last_activity_date", "String"),
    ("title", "String"),
    ("tags", "String"),
    ("content_license", "String"),
    ("closed_date", "String"),
    ("community_owned_date", "String"),
    ("deletion_date", "String"),
    ("text", "String"),
)


@dataclass(frozen=True)
class _SiteSpec:
    site: str
    archive_size: int
    sha1: str

    @property
    def source_url(self) -> str:
        return f"{_ARCHIVE_BASE_URL}/{self.site}.7z"


_SITE_SPECS = (
    _SiteSpec(
        site="quant.stackexchange.com",
        archive_size=60_845_478,
        sha1="6df03f8745978abb6d35ed8c6408fe2523c53a6b",
    ),
    _SiteSpec(
        site="money.stackexchange.com",
        archive_size=140_880_807,
        sha1="7513703962372537ef59417e30510174d343d116",
    ),
    _SiteSpec(
        site="economics.stackexchange.com",
        archive_size=47_102_959,
        sha1="ee27814be73ae0cbe5589426d1c05964aba5f3cb",
    ),
)
_SITE_SPECS_BY_NAME = MappingProxyType({spec.site: spec for spec in _SITE_SPECS})
_DEFAULT_SITES = tuple(spec.site for spec in _SITE_SPECS)


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"Required executable {name!r} was not found on PATH. "
            f"Install {name!r} before running the finance pipeline."
        )
    return executable


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_length_bounds(min_length: int, max_length: int) -> None:
    if (
        not isinstance(min_length, int)
        or isinstance(min_length, bool)
        or min_length < 0
    ):
        raise ValueError("min_length must be a nonnegative integer")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length <= min_length
    ):
        raise ValueError("max_length must be an integer greater than min_length")


def _select_site_specs(sites: Sequence[str] | None) -> tuple[_SiteSpec, ...]:
    selected_sites = _DEFAULT_SITES if sites is None else tuple(sites)
    if not selected_sites:
        raise ValueError("sites must contain at least one site")

    unknown_sites = [site for site in selected_sites if site not in _SITE_SPECS_BY_NAME]
    if unknown_sites:
        choices = ", ".join(_DEFAULT_SITES)
        unknown = ", ".join(repr(site) for site in unknown_sites)
        raise ValueError(
            f"Unknown finance site(s): {unknown}. Expected one of: {choices}"
        )

    if len(set(selected_sites)) != len(selected_sites):
        raise ValueError("sites must not contain duplicates")

    return tuple(_SITE_SPECS_BY_NAME[site] for site in selected_sites)


def _validate_directory_layout(
    work_dir: Path,
    output_dir: Path,
    selected_specs: tuple[_SiteSpec, ...],
) -> None:
    for spec in selected_specs:
        site_work_dir = work_dir / f"release={_RELEASE}" / f"site={spec.site}"
        final_dir, _ = _output_paths(output_dir, spec.site)
        if (
            site_work_dir == final_dir
            or site_work_dir in final_dir.parents
            or final_dir in site_work_dir.parents
        ):
            raise ValueError(
                f"Output for {spec.site} would be deleted with its work directory; "
                "choose a different work_dir or output_dir"
            )


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive(path: Path, spec: _SiteSpec) -> None:
    actual_size = path.stat().st_size
    if actual_size != spec.archive_size:
        raise RuntimeError(
            f"{path} has size {actual_size:,} bytes; expected "
            f"{spec.archive_size:,} bytes for {spec.site}. The file was retained."
        )

    actual_sha1 = _sha1(path)
    if actual_sha1 != spec.sha1:
        raise RuntimeError(
            f"{path} has SHA-1 {actual_sha1}; expected {spec.sha1} for "
            f"{spec.site}. The file was retained."
        )


def _download_archive(
    spec: _SiteSpec,
    site_work_dir: Path,
    curl_executable: str,
) -> Path:
    archive_path = site_work_dir / f"{spec.site}.7z"
    partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")

    if archive_path.exists():
        if not archive_path.is_file():
            raise RuntimeError(f"Expected archive path to be a file: {archive_path}")
        _LOGGER.info("Verifying existing archive for %s", spec.site)
        _verify_archive(archive_path, spec)
        return archive_path

    site_work_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Downloading %s", spec.source_url)
    command = [
        curl_executable,
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
        str(partial_path),
        spec.source_url,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Download failed for {spec.site}; resumable partial data was retained "
            f"at {partial_path}"
        ) from error

    _verify_archive(partial_path, spec)
    partial_path.replace(archive_path)
    return archive_path


def _extract_posts_xml(
    spec: _SiteSpec,
    archive_path: Path,
    site_work_dir: Path,
    seven_zip_executable: str,
) -> Path:
    extraction_dir = site_work_dir / "extracted"
    posts_path = extraction_dir / "Posts.xml"
    if posts_path.is_file() and posts_path.stat().st_size > 0:
        _LOGGER.info("Reusing extracted Posts.xml for %s", spec.site)
        return posts_path

    extraction_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Extracting Posts.xml for %s", spec.site)
    command = [
        seven_zip_executable,
        "e",
        "-y",
        "-aoa",
        f"-o{extraction_dir}",
        str(archive_path),
        "Posts.xml",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Failed to extract Posts.xml for {spec.site}; the verified archive "
            f"was retained at {archive_path}"
        ) from error

    if not posts_path.is_file() or posts_path.stat().st_size == 0:
        raise RuntimeError(
            f"The verified archive for {spec.site} did not produce a nonempty "
            f"root-level Posts.xml at {posts_path}"
        )
    return posts_path


def _processing_settings(
    batch_size: int,
    min_length: int,
    max_length: int,
) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "post_type_ids": [1, 2],
        "html_parser": "resiliparse",
        "html_input": "stackexchange_fragment_wrapped_as_document",
        "encoding_repair": "ftfy",
        "min_length": min_length,
        "max_length": max_length,
        "length_bounds": "exclusive",
        "parquet_compression": _PARQUET_COMPRESSION,
    }


def _expected_manifest_values(
    spec: _SiteSpec,
    processing_settings: dict[str, object],
) -> dict[str, object]:
    return {
        "manifest_version": _MANIFEST_VERSION,
        "release": _RELEASE,
        "site": spec.site,
        "source_url": spec.source_url,
        "sha1": spec.sha1,
        "archive_size": spec.archive_size,
        "processing": processing_settings,
        "schema": [
            {"name": column_name, "dtype": dtype}
            for column_name, dtype in _EXPECTED_SCHEMA
        ],
    }


def _load_matching_manifest(
    final_dir: Path,
    expected_values: dict[str, object],
) -> dict[str, object] | None:
    manifest_path = final_dir / "_manifest.json"
    success_path = final_dir / "_SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None

    if any(manifest.get(key) != value for key, value in expected_values.items()):
        return None

    row_count = manifest.get("row_count")
    shard_paths = manifest.get("shard_paths")
    if (
        not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count <= 0
        or not isinstance(shard_paths, list)
        or not shard_paths
        or any(not isinstance(path, str) for path in shard_paths)
    ):
        return None

    for relative_path in shard_paths:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            return None
        if not (final_dir / path).is_file():
            return None

    return manifest


def _output_paths(output_dir: Path, site: str) -> tuple[Path, Path]:
    site_output_dir = output_dir / f"release={_RELEASE}" / f"site={site}"
    return site_output_dir / "posts", site_output_dir / "posts.inprogress"


def _preflight_output(
    spec: _SiteSpec,
    output_dir: Path,
    expected_values: dict[str, object],
) -> bool:
    final_dir, staging_dir = _output_paths(output_dir, spec.site)
    if staging_dir.exists() or staging_dir.is_symlink():
        raise RuntimeError(
            f"Staging output already exists for {spec.site}: {staging_dir}. "
            "Inspect or move it before restarting; it will not be overwritten."
        )

    if not final_dir.exists() and not final_dir.is_symlink():
        return False
    if not final_dir.is_dir():
        raise RuntimeError(f"Final output path is not a directory: {final_dir}")
    if _load_matching_manifest(final_dir, expected_values) is None:
        raise RuntimeError(
            f"Final output for {spec.site} exists but is incomplete or does not "
            f"match this run: {final_dir}. It will not be overwritten."
        )
    return True


def _dataframe_schema(dataframe: daft.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, str(field.dtype)) for field in dataframe.schema())


def _verify_staged_parquet(staging_dir: Path) -> tuple[int, list[str]]:
    parquet_paths = sorted(staging_dir.rglob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No Parquet shards were written to {staging_dir}")

    dataframe = daft.read_parquet([str(path) for path in parquet_paths])
    actual_schema = _dataframe_schema(dataframe)
    if actual_schema != _EXPECTED_SCHEMA:
        raise RuntimeError(
            f"Staged Parquet at {staging_dir} has unexpected schema "
            f"{actual_schema!r}; expected {_EXPECTED_SCHEMA!r}"
        )

    row_count = dataframe.count_rows()
    if row_count <= 0:
        raise RuntimeError(f"Staged Parquet at {staging_dir} contains no rows")

    relative_paths = [
        path.relative_to(staging_dir).as_posix() for path in parquet_paths
    ]
    return row_count, relative_paths


def _build_engine(min_length: int, max_length: int) -> DataEngine:
    from llm_data.encoding import FixEncoding
    from llm_data.engine import DataEngine
    from llm_data.filters.filters import LengthFilter
    from llm_data.parsers.html import ParseHtml

    return (
        DataEngine(name="StackExchangeFinance")
        .add(ParseHtml(input_column="body", output_column="text"))
        .add(FixEncoding(input_column="text", output_column="text"))
        .add(
            LengthFilter(
                input_column="text",
                min_len=min_length,
                max_len=max_length,
            )
        )
    )


def _wrap_html_fragment(dataframe: daft.DataFrame) -> daft.DataFrame:
    return dataframe.with_column(
        _ORIGINAL_BODY_COLUMN,
        daft.col("body"),
    ).with_column(
        "body",
        daft.lit(_HTML_DOCUMENT_PREFIX)
        + daft.col("body")
        + daft.lit(_HTML_DOCUMENT_SUFFIX),
    )


def _restore_original_body(dataframe: daft.DataFrame) -> daft.DataFrame:
    return dataframe.with_column(
        "body",
        daft.col(_ORIGINAL_BODY_COLUMN),
    ).exclude(_ORIGINAL_BODY_COLUMN)


def _process_site(
    spec: _SiteSpec,
    work_dir: Path,
    output_dir: Path,
    batch_size: int,
    processing_settings: dict[str, object],
    expected_manifest_values: dict[str, object],
    curl_executable: str,
    seven_zip_executable: str,
) -> Path:
    final_dir, staging_dir = _output_paths(output_dir, spec.site)
    site_work_dir = work_dir / f"release={_RELEASE}" / f"site={spec.site}"

    if _load_matching_manifest(final_dir, expected_manifest_values) is not None:
        _LOGGER.info("Skipping completed site %s", spec.site)
        if site_work_dir.exists():
            shutil.rmtree(site_work_dir)
        return final_dir

    from llm_data.loader import StackExchangePostsLoader

    engine = _build_engine(
        min_length=int(processing_settings["min_length"]),
        max_length=int(processing_settings["max_length"]),
    )
    loader = StackExchangePostsLoader(site=spec.site, batch_size=batch_size)

    archive_path = _download_archive(spec, site_work_dir, curl_executable)
    posts_path = _extract_posts_xml(
        spec,
        archive_path,
        site_work_dir,
        seven_zip_executable,
    )

    staging_dir.mkdir(parents=True)

    batch_count = 0
    for dataframe in loader.read_data(str(posts_path)):
        batch_count += 1
        _LOGGER.info("Processing %s batch %d", spec.site, batch_count)
        processed = _restore_original_body(engine.run(_wrap_html_fragment(dataframe)))
        actual_schema = _dataframe_schema(processed)
        if actual_schema != _EXPECTED_SCHEMA:
            raise RuntimeError(
                f"Processed batch {batch_count} for {spec.site} has unexpected "
                f"schema {actual_schema!r}; expected {_EXPECTED_SCHEMA!r}"
            )
        processed.write_parquet(
            staging_dir,
            compression=_PARQUET_COMPRESSION,
            write_mode="append",
        )

    if batch_count == 0:
        raise RuntimeError(f"Posts.xml for {spec.site} contained no selected posts")

    row_count, shard_paths = _verify_staged_parquet(staging_dir)
    manifest = {
        **expected_manifest_values,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "shard_paths": shard_paths,
    }
    (staging_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging_dir / "_SUCCESS").write_text("", encoding="utf-8")

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.replace(final_dir)
    if _load_matching_manifest(final_dir, expected_manifest_values) is None:
        raise RuntimeError(
            f"Final output verification failed after publishing {final_dir}; "
            f"source files were retained at {site_work_dir}"
        )

    shutil.rmtree(site_work_dir)
    _LOGGER.info(
        "Completed %s with %d rows across %d shard(s)",
        spec.site,
        row_count,
        len(shard_paths),
    )
    return final_dir


def run_finance_pipeline(
    work_dir: str | Path,
    output_dir: str | Path,
    sites: Sequence[str] | None = None,
    batch_size: int = 50_000,
    min_length: int = 0,
    max_length: int = 10_000,
) -> list[Path]:
    _validate_positive_integer(batch_size, "batch_size")
    _validate_length_bounds(min_length, max_length)
    selected_specs = _select_site_specs(sites)

    resolved_work_dir = Path(work_dir).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    _validate_directory_layout(
        resolved_work_dir,
        resolved_output_dir,
        selected_specs,
    )

    curl_executable = _require_executable("curl")
    seven_zip_executable = _require_executable("7z")
    processing_settings = _processing_settings(
        batch_size,
        min_length,
        max_length,
    )

    expected_values = {
        spec.site: _expected_manifest_values(spec, processing_settings)
        for spec in selected_specs
    }
    completed = {
        spec.site: _preflight_output(
            spec,
            resolved_output_dir,
            expected_values[spec.site],
        )
        for spec in selected_specs
    }

    resolved_work_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for spec in selected_specs:
        if completed[spec.site]:
            _LOGGER.info("Found verified completed output for %s", spec.site)
        output_paths.append(
            _process_site(
                spec=spec,
                work_dir=resolved_work_dir,
                output_dir=resolved_output_dir,
                batch_size=batch_size,
                processing_settings=processing_settings,
                expected_manifest_values=expected_values[spec.site],
                curl_executable=curl_executable,
                seven_zip_executable=seven_zip_executable,
            )
        )

    return output_paths


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and process the March 31, 2026 finance Stack Exchange dumps."
        )
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=_DEFAULT_SITES,
        default=None,
        help="Sites to process in the supplied order (default: all three).",
    )
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=10_000)
    return parser

def qa_pairs(
    output_path: str | Path,
    min_score: int = 5,
) -> None:
    parquet_paths = sorted(Path(output_path).rglob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No Parquet shards found in {output_path}")
    df = daft.read_parquet([str(path) for path in parquet_paths])

    questions = df.where((col("post_type_id") == 1) & (col("score") >= min_score)).select(
        col("id").alias("question_id"),
        col("title"),
        col("text").alias("question_text"),
        col("score").alias("question_score"),
        col("accepted_answer_id"),
    )
    answers = df.where(col("post_type_id") == 2).select(
        col("id").alias("answer_id"),
        col("parent_id").alias("question_id"),
        col("text").alias("answer_text"),
        col("score").alias("answer_score"),
    )
    pairs = questions.join(answers, on="question_id").where(
        (col("answer_score") >= min_score)
        | (col("answer_id") == col("accepted_answer_id"))
    )
    pairs.write_parquet(
        str(Path(output_path).parent / "qa_pairs"),
        compression=_PARQUET_COMPRESSION,
        write_mode="overwrite",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        output_paths = run_finance_pipeline(
            work_dir=arguments.work_dir,
            output_dir=arguments.output_dir,
            sites=arguments.sites,
            batch_size=arguments.batch_size,
            min_length=arguments.min_length,
            max_length=arguments.max_length,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(status=1, message=f"error: {error}\n")

    for output_path in output_paths:
        qa_pairs(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
