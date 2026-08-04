import daft
import fasttext
from daft import DataFrame, col
from huggingface_hub import hf_hub_download


# Returns '__label__eng_Latn' for English
@daft.func
def language_prediction(txt: str | None, model_path: str) -> str:
    if txt is None:
        return ""

    model = fasttext.load_model(model_path)
    one_line = " ".join(txt.splitlines()).strip()

    if len(one_line) == 0:
        return ""

    return model.predict([one_line], k=1)[0][0][0]


# Language extraction with glotlid
class ExtractLanguage:
    def __init__(self, model_repo_id="cis-lmu/glotlid", model_filename="model.bin"):
        self.model_path = hf_hub_download(
            repo_id=model_repo_id, filename=model_filename
        )

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.with_column(
            "language", language_prediction(col("text"), self.model_path)
        )
        return df
