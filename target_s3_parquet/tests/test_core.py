"""Tests standard target features using the built-in SDK tests library."""

import pytest
from typing import Any, Dict
from unittest.mock import patch

from singer_sdk.testing import get_target_test_class

from target_s3_parquet.sinks import DEFAULT_SCHEMA_DESCRIPTION
from target_s3_parquet.target import TargetS3Parquet

SAMPLE_CONFIG: Dict[str, Any] = {
    "s3_path": "s3://mock-bucket/path",
    "athena_database": "mock_database",
}


# Run standard built-in target tests from the SDK:
StandardTargetTests = get_target_test_class(
    target_class=TargetS3Parquet,
    config=SAMPLE_CONFIG,
)


class TestTargetS3Parquet(StandardTargetTests):  # type: ignore[misc, valid-type]
    """Standard Target Tests."""

    @pytest.fixture(autouse=True)
    def mock_aws(self):
        with patch("awswrangler.catalog.create_database", return_value=None), patch(
            "awswrangler.catalog.does_table_exist", return_value=False
        ), patch("awswrangler.catalog.table", return_value=None), patch(
            "awswrangler.s3.to_parquet", return_value=None
        ):
            yield


def test_sink_creates_athena_database():
    with patch(
        "awswrangler.catalog.create_database", return_value=None
    ) as create_db, patch(
        "awswrangler.catalog.does_table_exist", return_value=False
    ), patch(
        "awswrangler.catalog.table", return_value=None
    ):
        target = TargetS3Parquet(config=SAMPLE_CONFIG)
        target.default_sink_class(
            target=target,
            stream_name="mock_stream",
            schema={"properties": {"id": {"type": "integer"}}},
            key_properties=None,
        )

    create_db.assert_called_once_with(
        name="mock_database",
        description=DEFAULT_SCHEMA_DESCRIPTION,
        exist_ok=True,
    )


def test_sink_prepends_schema_prefix_to_athena_database():
    config = {
        **SAMPLE_CONFIG,
        "schema_prefix": "prefixed",
    }

    with patch(
        "awswrangler.catalog.create_database", return_value=None
    ) as create_db, patch(
        "awswrangler.catalog.does_table_exist", return_value=False
    ) as does_table_exist, patch(
        "awswrangler.catalog.table", return_value=None
    ), patch(
        "awswrangler.s3.to_parquet", return_value=None
    ) as to_parquet:
        target = TargetS3Parquet(config=config)
        sink = target.default_sink_class(
            target=target,
            stream_name="mock_stream",
            schema={"properties": {"id": {"type": "integer"}}},
            key_properties=None,
        )
        sink.process_batch({"records": [{"id": 1}]})

    create_db.assert_called_once_with(
        name="prefixed_mock_database",
        description=DEFAULT_SCHEMA_DESCRIPTION,
        exist_ok=True,
    )
    assert does_table_exist.call_args.kwargs["database"] == "prefixed_mock_database"
    assert to_parquet.call_args.kwargs["database"] == "prefixed_mock_database"
    assert (
        to_parquet.call_args.kwargs["path"]
        == "s3://mock-bucket/path/mock_database/mock_stream"
    )


def test_sink_uses_custom_schema_description():
    config = {
        **SAMPLE_CONFIG,
        "schema_description": "Custom database description",
    }

    with patch(
        "awswrangler.catalog.create_database", return_value=None
    ) as create_db, patch(
        "awswrangler.catalog.does_table_exist", return_value=False
    ), patch(
        "awswrangler.catalog.table", return_value=None
    ):
        target = TargetS3Parquet(config=config)
        target.default_sink_class(
            target=target,
            stream_name="mock_stream",
            schema={"properties": {"id": {"type": "integer"}}},
            key_properties=None,
        )

    create_db.assert_called_once_with(
        name="mock_database",
        description="Custom database description",
        exist_ok=True,
    )


def test_process_batch_uses_athena_database_in_s3_path():
    with patch("awswrangler.catalog.create_database", return_value=None), patch(
        "awswrangler.catalog.does_table_exist", return_value=False
    ), patch("awswrangler.catalog.table", return_value=None), patch(
        "awswrangler.s3.to_parquet", return_value=None
    ) as to_parquet:
        target = TargetS3Parquet(config=SAMPLE_CONFIG)
        sink = target.default_sink_class(
            target=target,
            stream_name="mock_stream",
            schema={"properties": {"id": {"type": "integer"}}},
            key_properties=None,
        )
        sink.process_batch({"records": [{"id": 1}]})

    assert (
        to_parquet.call_args.kwargs["path"]
        == "s3://mock-bucket/path/mock_database/mock_stream"
    )
