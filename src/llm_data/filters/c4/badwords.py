import daft
import requests

from llm_data.config import BAD_WORDS_URL


def load_badwords(url: str = BAD_WORDS_URL) -> frozenset[str]:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return frozenset(w.strip().lower() for w in resp.text.splitlines() if w.strip())


def make_badwords_filter(badwords: frozenset[str]):

    @daft.func
    def has_badword(text: str) -> bool:
        return not set(text.lower().split()).isdisjoint(badwords)

    return has_badword
