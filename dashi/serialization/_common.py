import json
import math
from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

_DASHI_TYPE_KEY = "__dashi_type__"
_SPECIAL_FLOAT_KEY = "__dashi_special_float__"

_TYPE_NDARRAY = "ndarray"
_TYPE_DATAFRAME = "dataframe"
_TYPE_TIMESTAMP = "timestamp"
_TYPE_VARIABLE_TYPE = "variable_type"

__all__ = [
    "sanitize_for_json",
    "to_json",
    "from_json",
]


def sanitize_for_json(data: Any) -> Any:
    """Return a JSON-compatible representation while preserving Dashi metadata. Use if you need to serialize data to a
    JSON file but do not want to export using the ``to_json`` function"""
    return _encode_json_value(data)


def to_json(data: Any) -> str:
    """Convert a serialized Dashi payload to a standards-compliant JSON string."""
    payload = data if _is_serialized_dashi_payload(data) else _encode_json_value(data)
    try:
        return json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        if payload is data:
            try:
                return json.dumps(_encode_json_value(data), allow_nan=False)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"Unable to serialize Dashi payload to JSON: {exc}") from exc


def from_json(json_string: str) -> dict:
    """Parse a JSON string back into a Python dictionary with Dashi scalar metadata restored."""
    if not isinstance(json_string, str):
        raise TypeError("json_string must be a string.")

    return _parse_json_if_needed(json_string)


def _encode_json_value(data: Any) -> Any:
    if data is None:
        return None

    if isinstance(data, dict):
        return {
            _validate_json_key(key, "payload"): _encode_json_value(value)
            for key, value in data.items()
        }

    if isinstance(data, (list, tuple)):
        return [_encode_json_value(value) for value in data]

    if isinstance(data, np.ndarray):
        return {
            _DASHI_TYPE_KEY: _TYPE_NDARRAY,
            "dtype": str(data.dtype),
            "data": _encode_json_value(data.tolist()),
        }

    if isinstance(data, pd.DataFrame):
        return {
            _DASHI_TYPE_KEY: _TYPE_DATAFRAME,
            "columns": _encode_json_value(list(data.columns)),
            "index": _encode_json_value(list(data.index)),
            "data": _encode_json_value(data.to_numpy(dtype=object).tolist()),
            "dtypes": {str(column): dtype.name for column, dtype in data.dtypes.items()},
        }

    if isinstance(data, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(data)
        if pd.isna(timestamp):
            return {_DASHI_TYPE_KEY: _TYPE_TIMESTAMP, "value": None}
        return {_DASHI_TYPE_KEY: _TYPE_TIMESTAMP, "value": timestamp.isoformat()}

    if isinstance(data, pd.Series):
        return _encode_json_value(data.tolist())

    if isinstance(data, np.bool_):
        return bool(data)

    if isinstance(data, np.integer):
        return int(data)

    if isinstance(data, (float, np.floating)):
        value = float(data)
        if math.isnan(value):
            return {_SPECIAL_FLOAT_KEY: "nan"}
        if math.isinf(value):
            return {_SPECIAL_FLOAT_KEY: "inf" if value > 0 else "-inf"}
        return value

    try:
        if pd.isna(data):
            return {_SPECIAL_FLOAT_KEY: "nan"}
    except (TypeError, ValueError):
        pass

    return data


def _is_serialized_dashi_payload(data: Any) -> bool:
    if not isinstance(data, Mapping):
        return False

    if _DASHI_TYPE_KEY in data:
        return True

    return bool(data) and all(_is_serialized_dashi_payload(value) for value in data.values())


def _decode_json_value(data: Any) -> Any:
    if isinstance(data, list):
        return [_decode_json_value(value) for value in data]

    if not isinstance(data, dict):
        return data

    if _SPECIAL_FLOAT_KEY in data:
        value = data[_SPECIAL_FLOAT_KEY]
        if value == "nan":
            return np.nan
        if value == "inf":
            return np.inf
        if value == "-inf":
            return -np.inf
        raise ValueError(f"Unknown special float marker: {value!r}.")

    payload_type = data.get(_DASHI_TYPE_KEY)
    if payload_type == _TYPE_NDARRAY:
        _require_keys(data, ["data", "dtype"], "ndarray payload")
        values = _decode_json_value(data["data"])
        try:
            return np.array(values, dtype=data["dtype"])
        except (TypeError, ValueError):
            return np.array(values)

    if payload_type == _TYPE_DATAFRAME:
        _require_keys(data, ["columns", "data", "dtypes"], "DataFrame payload")
        columns = _decode_json_value(data["columns"])
        values = _decode_json_value(data["data"])
        frame = pd.DataFrame(values, columns=columns)
        _restore_dataframe_dtypes(frame, data.get("dtypes", {}))
        if "index" in data:
            index = _decode_json_value(data["index"])
            if index is not None and len(index) == len(frame):
                frame.index = index
        return frame

    if payload_type == _TYPE_TIMESTAMP:
        if data.get("value") is None:
            return pd.NaT
        try:
            return pd.Timestamp(data["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid timestamp value: {data.get('value')!r}.") from exc

    return {key: _decode_json_value(value) for key, value in data.items()}


def _encode_variable_type(variable_type: Any) -> dict | None:
    if variable_type is None:
        return None

    if isinstance(variable_type, str):
        return {_DASHI_TYPE_KEY: _TYPE_VARIABLE_TYPE, "kind": "str", "value": variable_type}

    if hasattr(variable_type, "name"):
        return {_DASHI_TYPE_KEY: _TYPE_VARIABLE_TYPE, "kind": "dtype", "value": variable_type.name}

    return {_DASHI_TYPE_KEY: _TYPE_VARIABLE_TYPE, "kind": "str", "value": str(variable_type)}


def _decode_variable_type(variable_type: Any) -> Any:
    if variable_type is None:
        return None

    if not isinstance(variable_type, dict):
        return variable_type

    if variable_type.get(_DASHI_TYPE_KEY) != _TYPE_VARIABLE_TYPE:
        return _decode_json_value(variable_type)

    _require_keys(variable_type, ["kind", "value"], "variable_type payload")
    value = variable_type["value"]
    if variable_type["kind"] == "str":
        return value
    if variable_type["kind"] == "dtype":
        try:
            return pd.api.types.pandas_dtype(value)
        except TypeError:
            return value
    raise ValueError(f"Unknown variable_type kind: {variable_type['kind']!r}.")


def _array_or_none(value: Any, field_name: str) -> np.ndarray | None:
    decoded = _decode_json_value(value)
    if decoded is None:
        return None
    if isinstance(decoded, np.ndarray):
        return decoded
    try:
        return np.array(decoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field_name!r} cannot be reconstructed as a NumPy array.") from exc


def _dataframe_or_none(value: Any, field_name: str) -> pd.DataFrame | None:
    decoded = _decode_json_value(value)
    if decoded is None:
        return None
    if isinstance(decoded, pd.DataFrame):
        return decoded
    try:
        return pd.DataFrame(decoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field_name!r} cannot be reconstructed as a pandas DataFrame.") from exc


def _parse_json_if_needed(data: dict | str) -> dict:
    parsed = data
    while isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid Dashi JSON payload.") from exc

    parsed = _decode_json_value(parsed)
    _require_mapping(parsed, "Dashi payload")
    return parsed


def _restore_dataframe_dtypes(frame: pd.DataFrame, dtypes: Mapping[str, str]) -> None:
    for column in frame.columns:
        dtype_name = dtypes.get(str(column))
        if dtype_name is None:
            continue

        try:
            if dtype_name == "category":
                frame[column] = frame[column].astype("category")
            elif dtype_name.startswith("datetime64"):
                frame[column] = pd.to_datetime(frame[column])
            elif dtype_name != "object":
                frame[column] = frame[column].astype(dtype_name)
        except (TypeError, ValueError):
            continue


def _require_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a dictionary-like object.")


def _require_keys(data: Mapping[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}.")


def _require_entry_value(entry: Any, context: str) -> Any:
    _require_mapping(entry, context)
    _require_keys(entry, ["label", "value"], context)
    return entry["value"]


def _decode_label(label: Any, context: str) -> Any:
    decoded_label = _decode_json_value(label)
    try:
        hash(decoded_label)
    except TypeError as exc:
        raise TypeError(f"{context} must be hashable after decoding.") from exc
    return decoded_label


def _validate_json_key(key: Any, context: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"All {context} dictionary keys must be strings for JSON serialization.")
    return key


def _validate_reconstructed_map(data_map: Any, context: str, check_method: Any | None = None) -> None:
    try:
        validation_result = check_method(data_map) if check_method is not None else data_map.check()
    except Exception as exc:
        raise ValueError(f"Reconstructed {context} failed validation: {exc}") from exc

    if validation_result is not True:
        raise ValueError(f"Reconstructed {context} is invalid: {'; '.join(validation_result)}")
