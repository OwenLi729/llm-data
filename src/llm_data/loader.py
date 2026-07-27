from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Optional

import daft
from daft import (
    CheckpointConfig,
    CheckpointStore,
    DataFrame,
    DataType,
    KeyFilteringSettings,
    Series,
)
from lxml import etree

from llm_data.utils import checkpoint_uri

daft.set_runner_ray()


class DataLoader:
    def __init__(
        self,
        loader_type: str = "parquet",
        checkpoint_path: Optional[str] = "data_checkpoints",
        file_path_column_name: Optional[str] = "source_path",
        checkpoint_on: Optional[str] = "source_path",
        num_workers: Optional[int] = None,
        cpus_per_worker: Optional[float] = None,
    ):
        self.loader_type = loader_type
        self.file_path_column_name = file_path_column_name

        if checkpoint_path and checkpoint_on:
            self.checkpoint_path = checkpoint_uri(checkpoint_path)
            self.config = CheckpointConfig(
                store=CheckpointStore(self.checkpoint_path),
                on=checkpoint_on,
                settings=KeyFilteringSettings(
                    num_workers=num_workers,
                    cpus_per_worker=cpus_per_worker,
                ),
            )

    def read_data(self, input_path: str):
        if self.loader_type == "parquet":
            df = daft.read_parquet(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "json":
            df = daft.read_json(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "csv":
            df = daft.read_csv(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "huggingface":
            df = daft.read_huggingface(input_path)
        elif self.loader_type == "lance":
            df = daft.read_lance(input_path, checkpoint=self.config)
        elif self.loader_type == "warc":
            df = daft.read_warc(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        else:
            raise ValueError("DataLoader not supported.")
        return df


@dataclass
class StackExchangePostsLoader:
    site: str
    batch_size: int = 50_000
    post_type_ids: tuple[int, ...] = (1, 2)

    _POST_INTEGER_FIELDS: ClassVar[tuple[tuple[str, str, bool], ...]] = (
        ("id", "Id", True),
        ("post_type_id", "PostTypeId", True),
        ("accepted_answer_id", "AcceptedAnswerId", False),
        ("parent_id", "ParentId", False),
        ("score", "Score", False),
        ("view_count", "ViewCount", False),
        ("owner_user_id", "OwnerUserId", False),
        ("last_editor_user_id", "LastEditorUserId", False),
        ("answer_count", "AnswerCount", False),
        ("comment_count", "CommentCount", False),
        ("favorite_count", "FavoriteCount", False),
    )

    _POST_TEXT_FIELDS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("creation_date", "CreationDate"),
        ("body", "Body"),
        ("owner_display_name", "OwnerDisplayName"),
        ("last_editor_display_name", "LastEditorDisplayName"),
        ("last_edit_date", "LastEditDate"),
        ("last_activity_date", "LastActivityDate"),
        ("title", "Title"),
        ("tags", "Tags"),
        ("content_license", "ContentLicense"),
        ("closed_date", "ClosedDate"),
        ("community_owned_date", "CommunityOwnedDate"),
        ("deletion_date", "DeletionDate"),
    )

    def __post_init__(self):
        if not isinstance(self.site, str) or not self.site.strip():
            raise ValueError("site must be a nonempty string")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not self.post_type_ids:
            raise ValueError("post_type_ids must not be empty")
        if any(
            not isinstance(post_type_id, int) or isinstance(post_type_id, bool)
            for post_type_id in self.post_type_ids
        ):
            raise ValueError("post_type_ids must contain only integers")

    @staticmethod
    def _parse_integer(
        attributes,
        field_name: str,
        required: bool,
        input_path: str,
        row_number: int,
    ) -> Optional[int]:
        value = attributes.get(field_name)
        if value is None:
            if required:
                raise ValueError(
                    f"{input_path}: row {row_number} is missing required field "
                    f"{field_name}"
                )
            return None

        try:
            return int(value)
        except ValueError as error:
            raise ValueError(
                f"{input_path}: row {row_number} has invalid integer field "
                f"{field_name}={value!r}"
            ) from error

    @classmethod
    def _post_dataframe(cls, records: list[dict]) -> DataFrame:
        columns = {
            column_name: Series.from_pylist(
                [record[column_name] for record in records],
                name=column_name,
                dtype=DataType.int64(),
            )
            for column_name, _, _ in cls._POST_INTEGER_FIELDS
        }
        columns.update(
            {
                column_name: Series.from_pylist(
                    [record[column_name] for record in records],
                    name=column_name,
                    dtype=DataType.string(),
                )
                for column_name in (
                    "site",
                    *(name for name, _ in cls._POST_TEXT_FIELDS),
                )
            }
        )
        return daft.from_pydict(columns)

    def read_data(self, input_path: str) -> Iterator[DataFrame]:
        selected_post_types = frozenset(self.post_type_ids)
        records = []
        row_number = 0

        try:
            context = etree.iterparse(
                input_path,
                events=("end",),
                tag="row",
                load_dtd=False,
                no_network=True,
                resolve_entities=False,
                recover=False,
                huge_tree=True,
            )
            for _, element in context:
                row_number += 1
                completed_records = None
                try:
                    post_type_id = self._parse_integer(
                        element.attrib,
                        "PostTypeId",
                        required=True,
                        input_path=input_path,
                        row_number=row_number,
                    )
                    if post_type_id in selected_post_types:
                        record = {"site": self.site}
                        for (
                            column_name,
                            field_name,
                            required,
                        ) in self._POST_INTEGER_FIELDS:
                            if field_name == "PostTypeId":
                                value = post_type_id
                            else:
                                value = self._parse_integer(
                                    element.attrib,
                                    field_name,
                                    required=required,
                                    input_path=input_path,
                                    row_number=row_number,
                                )
                            record[column_name] = value
                        for column_name, field_name in self._POST_TEXT_FIELDS:
                            record[column_name] = element.attrib.get(field_name)
                        records.append(record)

                        if len(records) >= self.batch_size:
                            completed_records = records
                            records = []
                finally:
                    element.clear(keep_tail=True)
                    while element.getprevious() is not None:
                        del element.getparent()[0]

                if completed_records is not None:
                    yield self._post_dataframe(completed_records)
        except etree.XMLSyntaxError as error:
            raise ValueError(f"{input_path}: malformed XML: {error}") from error

        if records:
            yield self._post_dataframe(records)
