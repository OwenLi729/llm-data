import json

import daft

from llm_data.cli import build_pipeline
from llm_data.encoding import FixEncoding
from llm_data.filters.filters import LengthFilter
from llm_data.parsers.html import ParseHtml


def test_parse_html_with_trafilatura():
    html = """
    <html>
      <head><title>Moon facts</title></head>
      <body>
        <nav>Home About Contact</nav>
        <article>
          <h1>The Moon</h1>
          <p>The Moon is Earth's natural satellite and completes an orbit
          approximately every twenty-seven days.</p>
          <p>Scientists study lunar rocks to understand the early Solar System.</p>
        </article>
      </body>
    </html>
    """
    df = daft.from_pydict({"html": [html]})

    result = ParseHtml(parser_type="trafilatura")(df).to_pydict()

    assert "natural satellite" in result["text"][0]
    assert "Home About Contact" not in result["text"][0]


def test_fix_encoding_repairs_text():
    df = daft.from_pydict({"text": ["FranÃ§ais"]})

    result = FixEncoding()(df).to_pydict()

    assert result["text"] == ["Français"]


def test_length_filter_keeps_text_inside_bounds():
    df = daft.from_pydict({"text": ["no", "useful", "far too long"]})

    result = LengthFilter(min_len=2, max_len=10)(df).to_pydict()

    assert result["text"] == ["useful"]


def test_pipeline_normalizes_and_filters_warc_records():
    useful_html = """
    <html><body><article><h1>Useful page</h1><p>
    This useful article contains enough readable text to pass the configured
    minimum length while still remaining short enough for this test.
    </p></article></body></html>
    """.encode()
    headers = json.dumps({"WARC-Identified-Content-Language": "eng"})
    df = daft.from_pydict(
        {
            "WARC-Type": ["response", "request"],
            "warc_content": [useful_html, b""],
            "WARC-Target-URI": ["https://example.com/page", "https://example.com"],
            "WARC-Record-ID": ["record-1", "record-2"],
            "WARC-Date": ["2026-07-19", "2026-07-19"],
            "warc_headers": [headers, headers],
            "source_path": ["example.warc.gz", "example.warc.gz"],
        }
    )

    result = build_pipeline(min_length=50, max_length=1_000).run(df).to_pydict()

    assert len(result["text"]) == 1
    assert result["url"] == ["https://example.com/page"]
    assert result["record_id"] == ["record-1"]
    assert result["language"] == ["eng"]


def test_pipeline_drops_unusable_content():
    df = daft.from_pydict(
        {
            "WARC-Type": ["response"],
            "warc_content": [b"\x81\x8d\x8f\x90"],
            "WARC-Target-URI": ["https://example.com"],
            "WARC-Record-ID": ["record-1"],
            "WARC-Date": ["2026-07-19"],
            "warc_headers": ["{}"],
            "source_path": ["example.warc.gz"],
        }
    )

    result = build_pipeline(min_length=0, max_length=1_000).run(df).to_pydict()

    assert result["text"] == []
