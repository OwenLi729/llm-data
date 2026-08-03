import daft
import fasttext

from typing import Optional
from daft import DataFrame, col
from daft.functions import hash
from huggingface_hub import hf_hub_download

from llm_data.utils import decode_html


class ExtractWarc:
    def __init__(
        self, 
        limit: Optional[int] = None,
        name: str = "ExtractWarc",
    ):
        self.limit = limit
        self.name = name    

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.where(col("WARC-Type") == "response")
        if self.limit is not None:
            df = df.limit(self.limit)
        df = df.with_column("warc_hash", hash(col("warc_content")))
        df = df.with_column("html", decode_html(col("warc_content")))
        df = df.drop_null(col("html"))
        df = df.with_column("url", col("WARC-Target-URI"))
        df = df.with_column("record_id", col("WARC-Record-ID"))
        df = df.with_column("crawl_date", col("WARC-Date"))
        return df.select(
            "warc_hash",
            "html",
            "url",
            "record_id",
            "crawl_date",
            "source_path",
        )


# Retruuns '__label__eng_Latn' for English
@daft.func
def language_prediction(txt: str, model_path: str) -> str:
    model = fasttext.load_model(model_path)
    return model.predict([txt.replace('\n', ' ').strip()])[0][0][0]


# Language extraction with glotlid
class ExtractLanguage:

    def __init__(self,
                 model_repo_id='cis-lmu/glotlid',
                 model_filename='model.bin'):
        self.model_path = hf_hub_download(repo_id=model_repo_id, 
                                          filename=model_filename)

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.with_column("language", language_prediction(col("text"), self.model_path))
        return df

