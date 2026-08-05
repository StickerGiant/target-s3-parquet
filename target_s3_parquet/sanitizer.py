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


def coerce_untyped_values(value, schema):
    """Stringify values sitting at schema positions that carry no type.

    generate_tap_schema maps an untyped schema (e.g. Zendesk's polymorphic
    tickets.custom_fields[].value, declared as `{}`) to a string column. The
    data at those positions is whatever the source put there -- bool for a
    checkbox field, int for a numeric one -- and pyarrow refuses to write a
    bool into a string column, so the values have to be coerced to match the
    type we declared. Containers are JSON-encoded rather than repr'd so the
    result stays parseable downstream.

    Typed positions are walked but left untouched.
    """
    if value is None:
        return None

    resolved = resolve_anyof(schema) if isinstance(schema, dict) else {}
    declared = resolved.get("type") if isinstance(resolved, dict) else None

    if declared is None:
        # No resolvable type -- generate_tap_schema declared this string.
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=_convert_decimal)
        return str(value)

    cleaned = get_valid_types(declared)

    if cleaned == "array" and isinstance(value, list):
        items = resolved.get("items", {})
        return [coerce_untyped_values(v, items) for v in value]

    if cleaned == "object" and isinstance(value, dict):
        props = resolved.get("properties", {})
        return {
            k: coerce_untyped_values(v, props[k]) if k in props else v
            for k, v in value.items()
        }

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
