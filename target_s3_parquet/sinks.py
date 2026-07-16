"""S3Parquet target sink class, which handles writing streams."""


from typing import Dict, List, Optional
import awswrangler as wr
from pandas import DataFrame
from singer_sdk import Target
from singer_sdk.sinks import BatchSink
import json
from target_s3_parquet.data_type_generator import (
    generate_tap_schema,
    generate_current_target_schema,
)
from target_s3_parquet.sanitizer import (
    get_specific_type_attributes,
    apply_json_dump_to_df,
    stringify_df,
    convert_nested_decimals,
)


from datetime import datetime

STARTED_AT = datetime.now()
DEFAULT_SCHEMA_DESCRIPTION = (
    'Automatically created by Singer target "target-s3-parquet".'
)


class S3ParquetSink(BatchSink):
    """S3Parquet target sink class."""

    def __init__(
        self,
        target: Target,
        stream_name: str,
        schema: Dict,
        key_properties: Optional[List[str]],
    ) -> None:
        super().__init__(target, stream_name, schema, key_properties)

        self._ensure_athena_database()
        self._glue_schema = self._get_glue_schema()

    def _athena_database_name(self):
        schema_prefix = self.config.get("schema_prefix") or ""
        if schema_prefix and not schema_prefix.endswith("_"):
            schema_prefix = f"{schema_prefix}_"

        return f"{schema_prefix}{self.config.get('athena_database')}"

    def _athena_database_description(self):
        return self.config.get("schema_description") or DEFAULT_SCHEMA_DESCRIPTION

    def _ensure_athena_database(self):
        wr.catalog.create_database(
            name=self._athena_database_name(),
            description=self._athena_database_description(),
            exist_ok=True,
        )

    def _get_glue_schema(self):

        catalog_params = {
            "database": self._athena_database_name(),
            "table": self.stream_name,
        }

        if wr.catalog.does_table_exist(**catalog_params):
            return wr.catalog.table(**catalog_params)
        else:
            return DataFrame()

    max_size = 10000  # Max records to write in one batch

    def process_batch(self, context: dict) -> None:
        """Write out any prepped records and return once fully written."""
        # Sample:
        # ------
        # client.upload(context["file_path"])  # Upload file
        # Path(context["file_path"]).unlink()  # Delete local copy

        df = DataFrame(context["records"])

        for column in df.columns:
            if df[column].dtype == "object":
                df[column] = df[column].map(convert_nested_decimals)

        df["_sdc_started_at"] = STARTED_AT.timestamp()

        current_schema = generate_current_target_schema(self._get_glue_schema())
        tap_schema = generate_tap_schema(
            self.schema["properties"], only_string=self.config.get("stringify_schema")
        )

        dtype = {**current_schema, **tap_schema}

        if self.config.get("stringify_schema"):
            attributes_names = get_specific_type_attributes(
                self.schema["properties"], "object"
            )
            df_transformed = apply_json_dump_to_df(df, attributes_names)
            df = stringify_df(df_transformed)

        self.logger.debug(f"DType Definition: {dtype}")

        full_path = f"{self.config.get('s3_path')}/{self.config.get('athena_database')}/{self.stream_name}"

        wr.s3.to_parquet(
            df=df,
            index=False,
            compression="gzip",
            dataset=True,
            path=full_path,
            database=self._athena_database_name(),
            table=self.stream_name,
            mode="append",
            partition_cols=["_sdc_started_at"],
            schema_evolution=True,
            dtype=dtype,
        )

        self.logger.info(f"Uploaded {len(context['records'])}")

        context["records"] = []
