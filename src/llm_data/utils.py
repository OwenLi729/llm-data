from pathlib import Path
from typing import Literal

from daft import DataType
from resiliparse.parse.encoding import bytes_to_str, detect_encoding


def decode_html(html):
    """
    OpenWebMath: https://arxiv.org/abs/2310.06786
    Decodes the html if possible.
    First try to decode with utf-8, then try to detect the encoding.
    """
    try:
        html = bytes_to_str(html, "utf-8")
    except Exception:
        encoding = detect_encoding(html)
        if encoding is None or encoding == "utf-8":
            return
        try:
            html = bytes_to_str(html, encoding)
        except Exception:
            return
    return html


def checkpoint_uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).resolve().as_uri()


def daft_dtype(
    precision: Literal["float32", "binary", "int8"] = "float32",
):
    if precision == "float32":
        daft_dtype = DataType.float32()
    elif precision == "binary":
        daft_dtype = DataType.binary()
    elif precision == "int8":
        daft_dtype = DataType.int8()
    else:
        raise ValueError(f"Daft datatype not supported: {precision}")
    return daft_dtype
