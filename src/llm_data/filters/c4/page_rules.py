"""Page-level C4 heuristics, exposed as Daft row-wise UDFs."""
import daft

from .line_rules import ends_with_terminal_punctuation, has_min_words, mentions_javascript

MIN_SENTENCES = 5


@daft.func
def clean_page(text: str) -> str | None:
    """Keep only lines passing the line-level rules; drop the page if too short."""
    kept = [
        line
        for line in text.splitlines()
        if has_min_words(line) and ends_with_terminal_punctuation(line) and not mentions_javascript(line)
    ]
    return "\n".join(kept) if len(kept) >= MIN_SENTENCES else None


@daft.func
def has_lorem_ipsum(text: str) -> bool:
    return "lorem ipsum" in text.lower()


@daft.func
def has_curly_brace(text: str) -> bool:
    return "{" in text
