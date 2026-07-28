# local orchestration of finance.py, should not be tracked or pushed to GitHub
# solely for local use

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from lxml import etree

from llm_data.pipelines.finance import _process_site, qa_pairs

_ARCHIVE_BASE_URL = (
    "https://archive.org/download/stackexchange_20260331/stackexchange_20260331"
)

# site : sha1 of the release .7z archive
_SITES: dict[str, str] = {
    "quant.stackexchange.com": "6df03f8745978abb6d35ed8c6408fe2523c53a6b",
    "money.stackexchange.com": "7513703962372537ef59417e30510174d343d116",
    "economics.stackexchange.com": "ee27814be73ae0cbe5589426d1c05964aba5f3cb",
}


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
            site_work = work / "release=20260331" / f"site={site}"
            posts_xml = _fetch_posts_xml(site, site_work, curl, seven_zip)
            path = _process_site(
                site,
                work,
                out_root,
                args.batch_size,
                args.min_length,
                args.max_length,
                posts_xml,
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
