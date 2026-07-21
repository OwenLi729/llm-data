# llm-data

## Setup

```bash
uv sync
```

Requires Python 3.12 or newer.

## Process a WARC file

Download the [example Common Crawl WARC](https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz) outside the repository, then run:

```bash
uv run llm-data-process-warc input.warc.gz output
```

By default, this processes 1,000 responses, extracts text with Trafilatura,
fixes encoding with ftfy, and keeps text between 200 and 100,000 characters.
These thresholds are provisional.

```bash
uv run llm-data-process-warc input.warc.gz output \
  --limit 5000 \
  --min-length 500 \
  --max-length 200000
```

The output contains `text`, `url`, `record_id`, `crawl_date`, `language`, and
`source_path`. Use `--runner ray` for Ray or `--checkpoint-path` for
checkpointing.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```