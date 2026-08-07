import daft
from daft import col

from .badwords import load_badwords, make_badwords_filter
from .dedupe import dedupe
from .page_rules import clean_page, has_curly_brace, has_lorem_ipsum


def filter_c4(df: daft.DataFrame, doc_id: str = "doc_id", text: str = "text") -> daft.DataFrame:
    has_badword = make_badwords_filter(load_badwords())

    # Dedup implemented because it's not a hash-based dedupe
    df = (
        df.with_column(text, clean_page(col(text)))
        .drop_null(text)
        .with_column("_lorem", has_lorem_ipsum(col(text)))
        .with_column("_brace", has_curly_brace(col(text)))
        .with_column("_badword", has_badword(col(text)))
        .where(~col("_lorem") & ~col("_brace") & ~col("_badword") & col("_english"))
        .exclude("_lorem", "_brace", "_badword", "_english")
    )
    return dedupe(df, doc_id=doc_id, text=text)
