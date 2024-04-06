from datetime import timedelta

import polars as pl
from hgraph import generator, TS_SCHEMA, TSB

from hgraph_db.data_source import _extract_schema, DATA_FRAME_SOURCE, DataStore, _converter


__all__ = ("data_frame_source",)


@generator(resolvers={TS_SCHEMA: _extract_schema})
def data_frame_source(
        dfs: type[DATA_FRAME_SOURCE], dt_col: str, offset: timedelta = timedelta()
) -> TSB[TS_SCHEMA]:
    """
    Iterates over the data_frame, returning an instance of TS_SCHEMA for each row in the table.
    null values are not ticked.
    """
    df: pl.DataFrame
    dfs_instance = DataStore.instance().get_data_source(dfs)
    dt_converter = _converter(dfs_instance.schema[dt_col])
    for df in dfs_instance.iter_frames():
        for value in df.iter_rows(named=True):
            dt = dt_converter(value.pop(dt_col))
            yield dt + offset, value
