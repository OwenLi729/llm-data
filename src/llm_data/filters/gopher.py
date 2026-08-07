import daft
from daft import DataFrame, DataType, col
from dolma.core.data_types import Document
from dolma.taggers.gopher import GopherTagger


GOPHER_SCORE_COLUMNS = (
    "fraction_of_characters_in_most_common_2grams",
    "fraction_of_characters_in_most_common_3grams",
    "fraction_of_characters_in_most_common_4grams",
    "fraction_of_characters_in_duplicate_5grams",
    "fraction_of_characters_in_duplicate_6grams",
    "fraction_of_characters_in_duplicate_7grams",
    "fraction_of_characters_in_duplicate_8grams",
    "fraction_of_characters_in_duplicate_9grams",
    "fraction_of_characters_in_duplicate_10grams",
    "character_count",
    "word_count",
    "median_word_length",
    "symbol_to_word_ratio",
    "fraction_of_words_with_alpha_character",
    "required_word_count",
    "fraction_of_lines_starting_with_bullet_point",
    "fraction_of_lines_ending_with_ellipsis",
    "fraction_of_duplicate_lines",
    "fraction_of_characters_in_duplicate_lines",
)

GOPHER_MIN_THRESHOLDS = {
    "word_count": 50,
    "median_word_length": 3,
    "fraction_of_words_with_alpha_character": 0.8,
    "required_word_count": 2,
}

GOPHER_MAX_THRESHOLDS = {
    "word_count": 100000,
    "median_word_length": 10,
    "symbol_to_word_ratio": 0.1,
    "fraction_of_lines_starting_with_bullet_point": 0.9,
    "fraction_of_lines_ending_with_ellipsis": 0.3,
    "fraction_of_duplicate_lines": 0.3,
    "fraction_of_characters_in_duplicate_lines": 0.3,
    "fraction_of_characters_in_most_common_2grams": 0.2,
    "fraction_of_characters_in_most_common_3grams": 0.18,
    "fraction_of_characters_in_most_common_4grams": 0.16,
    "fraction_of_characters_in_duplicate_5grams": 0.15,
    "fraction_of_characters_in_duplicate_6grams": 0.14,
    "fraction_of_characters_in_duplicate_7grams": 0.13,
    "fraction_of_characters_in_duplicate_8grams": 0.12,
    "fraction_of_characters_in_duplicate_9grams": 0.11,
    "fraction_of_characters_in_duplicate_10grams": 0.10,
}


@daft.func(
    return_dtype=DataType.struct(
        {name: DataType.float64() for name in GOPHER_SCORE_COLUMNS}
    ),
    unnest=True,
)
def gopher_scores(text: str | None) -> dict[str, float]:
    result = GopherTagger().predict(Document(source="", id="", text=text or ""))
    scores = dict.fromkeys(GOPHER_SCORE_COLUMNS, 0.0)
    scores.update({span.type: span.score for span in result.spans})
    return scores


class GopherFilter:
    def __init__(
        self,
        input_column: str = "text",
    ):
        self.input_column = input_column

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.select("*", gopher_scores(col(self.input_column)))
        predicate = daft.lit(True)
        for column, threshold in GOPHER_MIN_THRESHOLDS.items():
            predicate &= col(column) >= threshold
        for column, threshold in GOPHER_MAX_THRESHOLDS.items():
            predicate &= col(column) <= threshold
        return df.where(predicate)
