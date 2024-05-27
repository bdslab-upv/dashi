from datetime import datetime

from all_classes import DataTemporalMap


def trim_data_temporal_map(
        data_temporal_map: DataTemporalMap,
        start_date: datetime = None,
        end_date: datetime = None
) -> DataTemporalMap:
    if start_date is None:
        start_date = data_temporal_map.dates.min()
    else:
        start_date = data_temporal_map.dates[data_temporal_map.dates >= start_date].min()

    if end_date is None:
        end_date = data_temporal_map.dates.max()
    else:
        end_date = data_temporal_map.dates[data_temporal_map.dates <= end_date].max()

    start_index = data_temporal_map.dates.get_loc(start_date)
    end_index = data_temporal_map.dates.get_loc(end_date)

    data_temporal_map.probability_map = data_temporal_map.probability_map[start_index:end_index]
    data_temporal_map.counts_map = data_temporal_map.counts_map[start_index:end_index]
    data_temporal_map.dates = data_temporal_map.dates[start_index:end_index]

    return data_temporal_map
