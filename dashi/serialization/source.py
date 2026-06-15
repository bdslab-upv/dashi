from typing import Any, Mapping

from dashi.serialization._common import (
    _DASHI_TYPE_KEY,
    _array_or_none,
    _dataframe_or_none,
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
from dashi.unsupervised_characterization.data_source_map.data_source_map import (
    DataSourceMap,
    MultiVariateDataSourceMap,
)

_TYPE_DATA_SOURCE_MAP = "DataSourceMap"
_TYPE_MULTIVARIATE_DATA_SOURCE_MAP = "MultiVariateDataSourceMap"
_TYPE_CONDITIONAL_SOURCE_MAP = "ConditionalSourceMap"
_TYPE_CONDITIONAL_UNIVARIATE_SOURCE_MAP = "ConditionalUnivariateSourceMap"

__all__ = [
    "data_source_map_to_dict",
    "dict_to_data_source_map",
    "multivariate_data_source_map_to_dict",
    "dict_to_multivariate_data_source_map",
    "conditional_source_map_to_dict",
    "dict_to_conditional_source_map",
    "conditional_univariate_source_map_to_dict",
    "dict_to_conditional_univariate_source_map",
]


def data_source_map_to_dict(dsm: DataSourceMap | dict[str, DataSourceMap]) -> dict:
    """Serialize one ``DataSourceMap`` or a dictionary of them."""
    if isinstance(dsm, dict):
        return {
            _validate_json_key(key, "data_source_map_to_dict"): _single_data_source_map_to_dict(value)
            for key, value in dsm.items()
        }

    return _single_data_source_map_to_dict(dsm)


def dict_to_data_source_map(data: dict | str) -> DataSourceMap | dict[str, DataSourceMap]:
    """Deserialize one ``DataSourceMap`` or a dictionary of them."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "DataSourceMap payload")

    if parsed.get(_DASHI_TYPE_KEY) == _TYPE_DATA_SOURCE_MAP or "probability_map" in parsed:
        return _dict_to_single_data_source_map(parsed)

    if parsed.get(_DASHI_TYPE_KEY) is not None:
        raise ValueError(f"Expected DataSourceMap payload, got {parsed.get(_DASHI_TYPE_KEY)!r}.")

    return {
        key: _dict_to_single_data_source_map(value)
        for key, value in parsed.items()
    }


def multivariate_data_source_map_to_dict(mdsm: MultiVariateDataSourceMap) -> dict:
    """Serialize one ``MultiVariateDataSourceMap``."""
    if not isinstance(mdsm, MultiVariateDataSourceMap):
        raise TypeError("mdsm must be a MultiVariateDataSourceMap.")

    data = _single_data_source_map_to_dict(mdsm, allow_multivariate=True)
    data[_DASHI_TYPE_KEY] = _TYPE_MULTIVARIATE_DATA_SOURCE_MAP
    data.update({
        "multivariate_probability_map": _encode_json_value(mdsm.multivariate_probability_map),
        "multivariate_counts_map": _encode_json_value(mdsm.multivariate_counts_map),
        "multivariate_support": _encode_json_value(mdsm.multivariate_support),
    })
    return data


def dict_to_multivariate_data_source_map(data: dict | str) -> MultiVariateDataSourceMap:
    """Deserialize one ``MultiVariateDataSourceMap``."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "MultiVariateDataSourceMap payload")
    payload_type = parsed.get(_DASHI_TYPE_KEY)
    if payload_type not in (None, _TYPE_MULTIVARIATE_DATA_SOURCE_MAP):
        raise ValueError(f"Expected MultiVariateDataSourceMap payload, got {payload_type!r}.")

    _require_keys(
        parsed,
        [
            "probability_map",
            "counts_map",
            "sources",
            "support",
            "variable_name",
            "variable_type",
            "multivariate_probability_map",
            "multivariate_counts_map",
            "multivariate_support",
        ],
        "MultiVariateDataSourceMap payload",
    )

    mdsm = MultiVariateDataSourceMap(
        probability_map=_array_or_none(parsed["probability_map"], "probability_map"),
        counts_map=_array_or_none(parsed["counts_map"], "counts_map"),
        sources=_array_or_none(parsed["sources"], "sources"),
        support=_dataframe_or_none(parsed["support"], "support"),
        variable_name=parsed["variable_name"],
        variable_type=_decode_variable_type(parsed["variable_type"]),
        multivariate_probability_map=_array_or_none(
            parsed["multivariate_probability_map"], "multivariate_probability_map"
        ),
        multivariate_counts_map=_array_or_none(parsed["multivariate_counts_map"], "multivariate_counts_map"),
        multivariate_support=_array_or_none(parsed["multivariate_support"], "multivariate_support"),
    )
    _validate_reconstructed_source_map(mdsm, "MultiVariateDataSourceMap")
    return mdsm


def conditional_source_map_to_dict(conditional_dsm: dict[Any, MultiVariateDataSourceMap]) -> dict:
    """Serialize the result of ``estimate_conditional_data_source_map``."""
    if not isinstance(conditional_dsm, dict):
        raise TypeError("conditional_dsm must be a dictionary.")

    entries = []
    for label, map_obj in conditional_dsm.items():
        if not isinstance(map_obj, MultiVariateDataSourceMap):
            raise TypeError("All conditional source map values must be MultiVariateDataSourceMap objects.")
        entries.append({
            "label": _encode_json_value(label),
            "value": multivariate_data_source_map_to_dict(map_obj),
        })

    return {
        _DASHI_TYPE_KEY: _TYPE_CONDITIONAL_SOURCE_MAP,
        "entries": entries,
    }


def dict_to_conditional_source_map(data: dict | str) -> dict[Any, MultiVariateDataSourceMap]:
    """Deserialize the result of ``conditional_source_map_to_dict``."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "conditional source map payload")

    payload_type = parsed.get(_DASHI_TYPE_KEY)
    if payload_type == _TYPE_CONDITIONAL_SOURCE_MAP:
        _require_keys(parsed, ["entries"], "conditional source map payload")
        if not isinstance(parsed["entries"], list):
            raise TypeError("conditional source map entries must be a list.")

        conditional_maps = {}
        for entry in parsed["entries"]:
            value = _require_entry_value(entry, "conditional source map entry")
            label = _decode_label(entry["label"], "conditional source map label")
            conditional_maps[label] = dict_to_multivariate_data_source_map(value)
        return conditional_maps

    if payload_type is not None:
        raise ValueError(f"Expected ConditionalSourceMap payload, got {payload_type!r}.")

    return {
        label: dict_to_multivariate_data_source_map(map_data)
        for label, map_data in parsed.items()
    }


def conditional_univariate_source_map_to_dict(
        conditional_dsm: dict[Any, DataSourceMap | dict[str, DataSourceMap]]
) -> dict:
    """Serialize the result of ``estimate_conditional_univariate_data_source_map``."""
    if not isinstance(conditional_dsm, dict):
        raise TypeError("conditional_dsm must be a dictionary.")

    entries = []
    for label, map_or_maps in conditional_dsm.items():
        entries.append({
            "label": _encode_json_value(label),
            "value": _conditional_univariate_value_to_dict(map_or_maps),
        })

    return {
        _DASHI_TYPE_KEY: _TYPE_CONDITIONAL_UNIVARIATE_SOURCE_MAP,
        "entries": entries,
    }


def dict_to_conditional_univariate_source_map(
        data: dict | str
) -> dict[Any, DataSourceMap | dict[str, DataSourceMap]]:
    """Deserialize the result of ``conditional_univariate_source_map_to_dict``."""
    parsed = _parse_json_if_needed(data)
    _require_mapping(parsed, "conditional univariate source map payload")

    payload_type = parsed.get(_DASHI_TYPE_KEY)
    if payload_type == _TYPE_CONDITIONAL_UNIVARIATE_SOURCE_MAP:
        _require_keys(parsed, ["entries"], "conditional univariate source map payload")
        if not isinstance(parsed["entries"], list):
            raise TypeError("conditional univariate source map entries must be a list.")

        conditional_maps = {}
        for entry in parsed["entries"]:
            value = _require_entry_value(entry, "conditional univariate source map entry")
            label = _decode_label(entry["label"], "conditional univariate source map label")
            conditional_maps[label] = _dict_to_conditional_univariate_value(value)
        return conditional_maps

    if payload_type is not None:
        raise ValueError(f"Expected ConditionalUnivariateSourceMap payload, got {payload_type!r}.")

    return {
        label: _dict_to_conditional_univariate_value(map_or_maps)
        for label, map_or_maps in parsed.items()
    }


def _single_data_source_map_to_dict(dsm: DataSourceMap, allow_multivariate: bool = False) -> dict:
    if isinstance(dsm, MultiVariateDataSourceMap) and not allow_multivariate:
        raise TypeError(
            "dsm is a MultiVariateDataSourceMap. Use multivariate_data_source_map_to_dict instead."
        )
    if not isinstance(dsm, DataSourceMap):
        raise TypeError("dsm must be a DataSourceMap.")

    return {
        _DASHI_TYPE_KEY: _TYPE_DATA_SOURCE_MAP,
        "probability_map": _encode_json_value(dsm.probability_map),
        "counts_map": _encode_json_value(dsm.counts_map),
        "sources": _encode_json_value(dsm.sources),
        "support": _encode_json_value(dsm.support),
        "variable_name": dsm.variable_name,
        "variable_type": _encode_variable_type(dsm.variable_type),
    }


def _dict_to_single_data_source_map(data: Mapping[str, Any]) -> DataSourceMap:
    _require_mapping(data, "DataSourceMap payload")
    payload_type = data.get(_DASHI_TYPE_KEY)
    if payload_type not in (None, _TYPE_DATA_SOURCE_MAP):
        raise ValueError(f"Expected DataSourceMap payload, got {payload_type!r}.")
    _require_keys(
        data,
        ["probability_map", "counts_map", "sources", "support", "variable_name", "variable_type"],
        "DataSourceMap payload",
    )

    dsm = DataSourceMap(
        probability_map=_array_or_none(data["probability_map"], "probability_map"),
        counts_map=_array_or_none(data["counts_map"], "counts_map"),
        sources=_array_or_none(data["sources"], "sources"),
        support=_dataframe_or_none(data["support"], "support"),
        variable_name=data["variable_name"],
        variable_type=_decode_variable_type(data["variable_type"]),
    )
    _validate_reconstructed_source_map(dsm, "DataSourceMap")
    return dsm


def _conditional_univariate_value_to_dict(map_or_maps: DataSourceMap | dict[str, DataSourceMap]) -> dict:
    if isinstance(map_or_maps, MultiVariateDataSourceMap):
        raise TypeError(
            "conditional univariate source map values must be DataSourceMap objects, "
            "not MultiVariateDataSourceMap objects."
        )

    if isinstance(map_or_maps, DataSourceMap):
        return _single_data_source_map_to_dict(map_or_maps)

    if isinstance(map_or_maps, dict):
        return {
            _validate_json_key(variable_name, "conditional univariate source map variable"):
                _single_data_source_map_to_dict(data_source_map)
            for variable_name, data_source_map in map_or_maps.items()
        }

    raise TypeError(
        "conditional univariate source map values must be DataSourceMap objects or dictionaries of them."
    )


def _dict_to_conditional_univariate_value(data: Mapping[str, Any]) -> DataSourceMap | dict[str, DataSourceMap]:
    _require_mapping(data, "conditional univariate source map value")
    payload_type = data.get(_DASHI_TYPE_KEY)
    if payload_type == _TYPE_DATA_SOURCE_MAP or "probability_map" in data:
        return _dict_to_single_data_source_map(data)

    if payload_type is not None:
        raise ValueError(f"Expected DataSourceMap payload or variable map dictionary, got {payload_type!r}.")

    return {
        variable_name: _dict_to_single_data_source_map(data_source_map)
        for variable_name, data_source_map in data.items()
    }


def _validate_reconstructed_source_map(data_map: DataSourceMap, context: str) -> None:
    if isinstance(data_map, MultiVariateDataSourceMap):
        _validate_reconstructed_map(data_map, context, check_method=DataSourceMap.check)
    else:
        _validate_reconstructed_map(data_map, context)
