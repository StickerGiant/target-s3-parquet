from pandas import DataFrame
import numpy as np
import json
from typing import List
from decimal import Decimal


def _remove_nulls(array):
    return [v for v in array if v != "null"]


def _convert_decimal(value):
    if isinstance(value, Decimal):
        return str(value)
    return value


def _deep_convert_decimal(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_deep_convert_decimal(v) for v in value]
    if isinstance(value, dict):
        return {k: _deep_convert_decimal(v) for k, v in value.items()}
    return value


def convert_nested_decimals(value):
    """Convert Decimals nested inside lists/dicts to float.

    pyarrow coerces a top-level Decimal column via its dtype hint, but builds
    nested struct/array columns straight from Python objects and rejects the
    Decimals a singer.decimal field yields. Scalars are returned untouched so
    the working top-level path keeps its existing dtype handling.
    """
    if isinstance(value, (list, dict)):
        return _deep_convert_decimal(value)
    return value


def get_valid_types(types):
    if isinstance(types, list):
        return _remove_nulls(types)[0]
    else:
        return types


def resolve_anyof(attributes):
    """Return the branch of an anyOf that carries the real type.

    Taps express a nullable field either as {"type": ["null", "array"], ...} or
    as {"anyOf": [{"type": "array", ...}, {"type": "null"}]}. In the latter,
    "items"/"properties" live on the branch, not the wrapper, so callers must
    read them from the value returned here. Returns attributes unchanged when
    there is no anyOf to resolve.
    """
    if "type" in attributes or not attributes.get("anyOf"):
        return attributes

    return next(
        (
            branch
            for branch in attributes["anyOf"]
            if branch.get("type") not in (None, "null")
        ),
        attributes,
    )


def type_from_anyof(attributes):
    resolved = resolve_anyof(attributes)
    return None if resolved is attributes else resolved.get("type")


def get_specific_type_attributes(schema: dict, attr_type: str) -> list:
    attributes_names = []
    for name, attributes in schema.items():
        attribute_type = attributes.get("type") or type_from_anyof(attributes)
        if attribute_type is None:
            raise Exception(f"Invalid schema format: {schema}")
        cleaned_type = get_valid_types(attribute_type)
        if cleaned_type == attr_type:
            attributes_names.append(name)
    return attributes_names


def get_valid_attributes(attributes_names: List[str], df: DataFrame) -> List:
    valid_attributes = attributes_names
    if len(attributes_names) > 0:
        valid_attributes = [
            attribute for attribute in attributes_names if attribute in df.columns
        ]
    return valid_attributes


def apply_json_dump_to_df(
    source_df: DataFrame, attributes_names: List[str]
) -> DataFrame:
    df = source_df.copy()
    valid_attributes = get_valid_attributes(attributes_names, df)
    if len(valid_attributes) > 0:
        for attribute in valid_attributes:
            df.loc[:, attribute] = df[attribute].apply(
                lambda x: json.dumps(x, default=_convert_decimal)
            )
    return df


def stringify_df(df: DataFrame) -> DataFrame:
    return df.fillna("NULL").astype(str).replace("NULL", np.nan)
