"""Line-level C4 heuristics."""

_TERMINAL_PUNCTUATION = (".", "!", "?", '"', "”")


def ends_with_terminal_punctuation(line: str) -> bool:
    """Line ends in a period, exclamation mark, question mark, or end quote."""
    return line.rstrip().endswith(_TERMINAL_PUNCTUATION)


def has_min_words(line: str, min_words: int = 3) -> bool:
    return len(line.split()) >= min_words


def mentions_javascript(line: str) -> bool:
    return "javascript" in line.lower()
