from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from dashi.serialization._common import (
    _DASHI_TYPE_KEY,
    _array_or_none,
    _dataframe_or_none,
    _decode_json_value,
    _decode_label,
    _decode_variable_type,
    _encode_json_value,
    _encode_variable_type,
    _parse_json_if_needed,
    _require_entry_value,
    _require_keys,
    _require_mapping,
    _validate_json_key,
    _validate_reconstructed_map,
)
from dashi.unsupervised_characterization.data_temporal_map.data_temporal_map import (
    DataTemporalMap,
    MultiVariateDataTemporalMap,
)

_TYPE_DATA_TEMPORAL_MAP = "DataTemporalMap"
_TYPE_MULTIVARIATE_DATA_TEMPORAL_MAP = "MultiVariateDataTemporalMap"
_TYPE_CONDITIONAL_TEMPORAL_MAP = "ConditionalTemporalMap"
_TYPE_CONDITIONAL_UNIVARIATE_TEMPORAL_MAP = "ConditionalUnivariateTemporalMap"

__all__ = [
    "data_temporal_map_to_dict",
    "dict_to_data_temporal_map",
    "multivariate_data_temporal_map_to_dict",
    "dict_to_multivariate_data_temporal_map",
    "conditional_temporal_map_to_dict",
    "dict_to_conditional_temporal_map",
    "conditional_univariate_temporal_map_to_dict",
    "dict_to_conditional_univariate_temporal_map",
]


def data_temporal_map_to_dict(dtm: DataTemporalMap | dict[str, DataTemporalMap]) -> dict:
    """Serialize one ``DataTemporalMap`` or a dictionary of them."""
    if isinstance(dtm, dict):
        return {
            _validate_json_key(key, "data_temporal_map_to_dict"): _single_data_temporal_map_to_dict(value)
            for key, value in dtm.items()
        }

    return _single_data_temporal_map_to_dict(dtm)


def dict_to_data_temporal_map(data: dict | str) -> DataTemporalMap | dict[str, DataTemporalMap]:
    """Deserialize one ``DataTemporalMap`` or a dictionary of them."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "DataTemporalMap payload")

    if parsed.get(_DASHI_TYPE_KEY) == _TYPE_DATA_TEMPORAL_MAP or "probability_map" in parsed:
        return _dict_to_single_data_temporal_map(parsed)

    if parsed.get(_DASHI_TYPE_KEY) is not None:
        raise ValueError(f"Expected DataTemporalMap payload, got {parsed.get(_DASHI_TYPE_KEY)!r}.")

    return {
        key: _dict_to_single_data_temporal_map(value)
        for key, value in parsed.items()
    }


def multivariate_data_temporal_map_to_dict(mdtm: MultiVariateDataTemporalMap) -> dict:
    """Serialize one ``MultiVariateDataTemporalMap``."""
    if not isinstance(mdtm, MultiVariateDataTemporalMap):
        raise TypeError("mdtm must be a MultiVariateDataTemporalMap.")

    data = _single_data_temporal_map_to_dict(mdtm, allow_multivariate=True)
    data[_DASHI_TYPE_KEY] = _TYPE_MULTIVARIATE_DATA_TEMPORAL_MAP
    data.update({
        "multivariate_probability_map": _encode_json_value(mdtm.multivariate_probability_map),
        "multivariate_counts_map": _encode_json_value(mdtm.multivariate_counts_map),
        "multivariate_support": _encode_json_value(mdtm.multivariate_support),
    })
    return data


def dict_to_multivariate_data_temporal_map(data: dict | str) -> MultiVariateDataTemporalMap:
    """Deserialize one ``MultiVariateDataTemporalMap``."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "MultiVariateDataTemporalMap payload")
    payload_type = parsed.get(_DASHI_TYPE_KEY)
    if payload_type not in (None, _TYPE_MULTIVARIATE_DATA_TEMPORAL_MAP):
        raise ValueError(f"Expected MultiVariateDataTemporalMap payload, got {payload_type!r}.")

    _require_keys(
        parsed,
        [
            "probability_map",
            "counts_map",
            "dates",
            "support",
            "variable_name",
            "variable_type",
            "period",
            "multivariate_probability_map",
            "multivariate_counts_map",
            "multivariate_support",
        ],
        "MultiVariateDataTemporalMap payload",
    )

    mdtm = MultiVariateDataTemporalMap(
        probability_map=_array_or_none(parsed["probability_map"], "probability_map"),
        counts_map=_array_or_none(parsed["counts_map"], "counts_map"),
        dates=_datetime_index_or_none(parsed["dates"], "dates"),
        support=_dataframe_or_none(parsed["support"], "support"),
        variable_name=parsed["variable_name"],
        variable_type=_decode_variable_type(parsed["variable_type"]),
        period=parsed["period"],
        multivariate_probability_map=_array_or_none(
            parsed["multivariate_probability_map"], "multivariate_probability_map"
        ),
        multivariate_counts_map=_array_or_none(parsed["multivariate_counts_map"], "multivariate_counts_map"),
        multivariate_support=_array_or_none(parsed["multivariate_support"], "multivariate_support"),
    )
    _validate_reconstructed_temporal_map(mdtm, "MultiVariateDataTemporalMap")
    return mdtm


def conditional_temporal_map_to_dict(conditional_dtm: dict[Any, MultiVariateDataTemporalMap]) -> dict:
    """Serialize a conditional temporal map dictionary returned by Dashi."""
    if not isinstance(conditional_dtm, dict):
        raise TypeError("conditional_dtm must be a dictionary.")

    entries = []
    for label, map_obj in conditional_dtm.items():
        if not isinstance(map_obj, MultiVariateDataTemporalMap):
            raise TypeError("All conditional temporal map values must be MultiVariateDataTemporalMap objects.")
        entries.append({
            "label": _encode_json_value(label),
            "value": multivariate_data_temporal_map_to_dict(map_obj),
        })

    return {
        _DASHI_TYPE_KEY: _TYPE_CONDITIONAL_TEMPORAL_MAP,
        "entries": entries,
    }


def dict_to_conditional_temporal_map(data: dict | str) -> dict[Any, MultiVariateDataTemporalMap]:
    """Deserialize a conditional temporal map dictionary."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "conditional temporal map payload")

    payload_type = parsed.get(_DASHI_TYPE_KEY)
    if payload_type == _TYPE_CONDITIONAL_TEMPORAL_MAP:
        _require_keys(parsed, ["entries"], "conditional temporal map payload")
        if not isinstance(parsed["entries"], list):
            raise TypeError("conditional temporal map entries must be a list.")

        conditional_maps = {}
        for entry in parsed["entries"]:
            value = _require_entry_value(entry, "conditional temporal map entry")
            label = _decode_label(entry["label"], "conditional temporal map label")
            conditional_maps[label] = dict_to_multivariate_data_temporal_map(value)
        return conditional_maps

    if payload_type is not None:
        raise ValueError(f"Expected ConditionalTemporalMap payload, got {payload_type!r}.")

    return {
        label: dict_to_multivariate_data_temporal_map(map_data)
        for label, map_data in parsed.items()
    }


def conditional_univariate_temporal_map_to_dict(
        conditional_dtm: dict[Any, DataTemporalMap | dict[str, DataTemporalMap]]
) -> dict:
    """Serialize the result of ``estimate_conditional_univariate_data_temporal_map``."""
    if not isinstance(conditional_dtm, dict):
        raise TypeError("conditional_dtm must be a dictionary.")

    entries = []
    for label, map_or_maps in conditional_dtm.items():
        entries.append({
            "label": _encode_json_value(label),
            "value": _conditional_univariate_value_to_dict(map_or_maps),
        })

    return {
        _DASHI_TYPE_KEY: _TYPE_CONDITIONAL_UNIVARIATE_TEMPORAL_MAP,
        "entries": entries,
    }


def dict_to_conditional_univariate_temporal_map(
        data: dict | str
) -> dict[Any, DataTemporalMap | dict[str, DataTemporalMap]]:
    """Deserialize the result of ``conditional_univariate_temporal_map_to_dict``."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "conditional univariate temporal map payload")

    payload_type = parsed.get(_DASHI_TYPE_KEY)
    if payload_type == _TYPE_CONDITIONAL_UNIVARIATE_TEMPORAL_MAP:
        _require_keys(parsed, ["entries"], "conditional univariate temporal map payload")
        if not isinstance(parsed["entries"], list):
            raise TypeError("conditional univariate temporal map entries must be a list.")

        conditional_maps = {}
        for entry in parsed["entries"]:
            value = _require_entry_value(entry, "conditional univariate temporal map entry")
            label = _decode_label(entry["label"], "conditional univariate temporal map label")
            conditional_maps[label] = _dict_to_conditional_univariate_value(value)
        return conditional_maps

    if payload_type is not None:
        raise ValueError(f"Expected ConditionalUnivariateTemporalMap payload, got {payload_type!r}.")

    return {
        label: _dict_to_conditional_univariate_value(map_or_maps)
        for label, map_or_maps in parsed.items()
    }


def _single_data_temporal_map_to_dict(dtm: DataTemporalMap, allow_multivariate: bool = False) -> dict:
    if isinstance(dtm, MultiVariateDataTemporalMap) and not allow_multivariate:
        raise TypeError(
            "dtm is a MultiVariateDataTemporalMap. Use multivariate_data_temporal_map_to_dict instead."
        )
    if not isinstance(dtm, DataTemporalMap):
        raise TypeError("dtm must be a DataTemporalMap.")

    return {
        _DASHI_TYPE_KEY: _TYPE_DATA_TEMPORAL_MAP,
        "probability_map": _encode_json_value(dtm.probability_map),
        "counts_map": _encode_json_value(dtm.counts_map),
        "dates": _encode_json_value(list(dtm.dates) if dtm.dates is not None else None),
        "support": _encode_json_value(dtm.support),
        "variable_name": dtm.variable_name,
        "variable_type": _encode_variable_type(dtm.variable_type),
        "period": dtm.period,
    }


def _dict_to_single_data_temporal_map(data: Mapping[str, Any]) -> DataTemporalMap:
    _require_mapping(data, "DataTemporalMap payload")
    payload_type = data.get(_DASHI_TYPE_KEY)
    if payload_type not in (None, _TYPE_DATA_TEMPORAL_MAP):
        raise ValueError(f"Expected DataTemporalMap payload, got {payload_type!r}.")
    _require_keys(
        data,
        ["probability_map", "counts_map", "dates", "support", "variable_name", "variable_type", "period"],
        "DataTemporalMap payload",
    )

    dtm = DataTemporalMap(
        probability_map=_array_or_none(data["probability_map"], "probability_map"),
        counts_map=_array_or_none(data["counts_map"], "counts_map"),
        dates=_datetime_index_or_none(data["dates"], "dates"),
        support=_dataframe_or_none(data["support"], "support"),
        variable_name=data["variable_name"],
        variable_type=_decode_variable_type(data["variable_type"]),
        period=data["period"],
    )
    _validate_reconstructed_temporal_map(dtm, "DataTemporalMap")
    return dtm


def _conditional_univariate_value_to_dict(map_or_maps: DataTemporalMap | dict[str, DataTemporalMap]) -> dict:
    if isinstance(map_or_maps, MultiVariateDataTemporalMap):
        raise TypeError(
            "conditional univariate temporal map values must be DataTemporalMap objects, "
            "not MultiVariateDataTemporalMap objects."
        )

    if isinstance(map_or_maps, DataTemporalMap):
        return _single_data_temporal_map_to_dict(map_or_maps)

    if isinstance(map_or_maps, dict):
        return {
            _validate_json_key(variable_name, "conditional univariate temporal map variable"):
                _single_data_temporal_map_to_dict(data_temporal_map)
            for variable_name, data_temporal_map in map_or_maps.items()
        }

    raise TypeError(
        "conditional univariate temporal map values must be DataTemporalMap objects or dictionaries of them."
    )


def _dict_to_conditional_univariate_value(data: Mapping[str, Any]) -> DataTemporalMap | dict[str, DataTemporalMap]:
    _require_mapping(data, "conditional univariate temporal map value")
    payload_type = data.get(_DASHI_TYPE_KEY)
    if payload_type == _TYPE_DATA_TEMPORAL_MAP or "probability_map" in data:
        return _dict_to_single_data_temporal_map(data)

    if payload_type is not None:
        raise ValueError(f"Expected DataTemporalMap payload or variable map dictionary, got {payload_type!r}.")

    return {
        variable_name: _dict_to_single_data_temporal_map(data_temporal_map)
        for variable_name, data_temporal_map in data.items()
    }


def _datetime_index_or_none(value: Any, field_name: str) -> pd.DatetimeIndex | None:
    decoded = _decode_json_value(value)
    if decoded is None:
        return None
    try:
        return pd.DatetimeIndex(pd.to_datetime(decoded))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field_name!r} cannot be reconstructed as a DatetimeIndex.") from exc


def _validate_reconstructed_temporal_map(data_map: DataTemporalMap, context: str) -> None:
    if isinstance(data_map, MultiVariateDataTemporalMap):
        # The current multivariate check does not match the shape produced by Dashi's own
        # multivariate temporal maps, so validate the shared DataTemporalMap fields here.
        _validate_reconstructed_map(data_map, context, check_method=DataTemporalMap.check)
    else:
        _validate_reconstructed_map(data_map, context)
