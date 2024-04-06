from abc import abstractmethod, ABC
from datetime import date, datetime, time, timedelta
from typing import Iterator, TypeVar, Optional, OrderedDict, Callable

import polars as pl
from hgraph import TSB, TS_SCHEMA, generator, ts_schema, HgTimeSeriesTypeMetaData, TS
from polars.datatypes.classes import FloatType, IntegerType, String, Boolean, Date, Datetime, Time, Duration, \
    Categorical, Array, Object, List

__all__ = ('DataFrameSource', 'DataStore', 'DATA_FRAME_SOURCE', 'data_frame_source')


class DataFrameSource(ABC):
    """
    Provide an abstraction over retrieving a data-source.
    This provides the ability to test the retrieval of data independent of
    the graph. This can then be provided to the ``data_frame_source`` generator
    to feed into the graph.
    """

    @property
    @abstractmethod
    def data_frame(self) -> pl.DataFrame:
        """
        Returns a data-frame representing this data source.
        """

    @property
    def schema(self) -> OrderedDict[str, pl.DataType]:
        """
        The schema describing this data source. By default, the code will get then data_frame and then
        extract the schema from that. If the data-source is large it is possible to provide the value
        directly. (Override the property)
        """
        df = self.data_frame
        return df.schema

    def iter_frames(self) -> Iterator[pl.DataFrame]:
        """
        Return the data source as a sequence of dataframes.
        By default, this is just an iterator over the data_frame provided by this data source.
        When possible, this is useful when the data source can return results in batches.
        Could produce better memory consumption and possibly improve the performance of the
        data source when back-testing.
        """
        return iter([self.data_frame])


DATA_FRAME_SOURCE = TypeVar("DATA_FRAME_SOURCE", bound=DataFrameSource)


class DataStore:
    """A cache of DataFrameSource instances"""

    _instance: Optional["DataStore"] = None

    def __init__(self):
        self._data_frame_sources: dict[str, DATA_FRAME_SOURCE] = {}

    def register_instance(self):
        if DataStore._instance is None:
            DataStore._instance = self
        else:
            raise RuntimeError("Datastore already registered")

    @staticmethod
    def release_instance():
        DataStore._instance = None

    @staticmethod
    def instance() -> "DataStore":
        if DataStore._instance is None:
            DataStore().register_instance()
        return DataStore._instance

    def set_data_source(self, dfs: type[DATA_FRAME_SOURCE], dfs_instance: DATA_FRAME_SOURCE):
        """
        Allow for pre-setting the data source. Useful when the data source requires initialisation.
        """
        self._data_frame_sources[dfs] = dfs_instance

    def get_data_source(self, dfs: type[DATA_FRAME_SOURCE]) -> DATA_FRAME_SOURCE:
        """
        Returns an instance of the DataFrameSource, if one exists, otherwise it will instantiate the
        data source, cache it and then return it.
        """
        dfs_instance = self._data_frame_sources.get(dfs)
        if dfs_instance is None:
            dfs_instance = dfs()
            self._data_frame_sources[dfs] = dfs_instance
        return dfs_instance


def _extract_schema(mapping, scalars) -> TS_SCHEMA:
    """Extract the schema from the mapping"""
    dfs: type[DATA_FRAME_SOURCE] = mapping[DATA_FRAME_SOURCE].py_type
    dfs_instance = DataStore.instance().get_data_source(dfs)
    dt_col = scalars["dt_col"]
    schema = dfs_instance.schema
    return ts_schema(**{k: _convert_type(v) for k, v in schema.items() if k != dt_col})


def _convert_type(pl_type: pl.DataType) -> HgTimeSeriesTypeMetaData:
    if isinstance(pl_type, IntegerType):
        return HgTimeSeriesTypeMetaData.parse_type(TS[int])
    if isinstance(pl_type, FloatType):
        return HgTimeSeriesTypeMetaData.parse_type(TS[float])
    if isinstance(pl_type, String):
        return HgTimeSeriesTypeMetaData.parse_type(TS[str])
    if isinstance(pl_type, Boolean):
        return HgTimeSeriesTypeMetaData.parse_type(TS[bool])
    if isinstance(pl_type, Date):
        return HgTimeSeriesTypeMetaData.parse_type(TS[date])
    if isinstance(pl_type, Datetime):
        return HgTimeSeriesTypeMetaData.parse_type(TS[datetime])
    if isinstance(pl_type, Time):
        return HgTimeSeriesTypeMetaData.parse_type(TS[time])
    if isinstance(pl_type, Duration):
        return HgTimeSeriesTypeMetaData.parse_type(TS[timedelta])
    if isinstance(pl_type, Categorical):
        return HgTimeSeriesTypeMetaData.parse_type(TS[str])
    if isinstance(pl_type, (List, Array)):
        tp: List = pl_type
        return HgTimeSeriesTypeMetaData.parse_type(TS[_convert_type(tp.inner).py_type])
    if isinstance(pl_type, Object):
        return HgTimeSeriesTypeMetaData.parse_type(TS[object])
    # Do Struct, still

    raise ValueError(f"Unable to convert {pl_type} to HgTimeSeriesTypeMetaData")


DT_COL_INDEX = TypeVar("DT_COL_INDEX", bound=int)


def _converter(dt_tp: pl.DataType) -> Callable[[date | datetime], datetime]:
    if isinstance(dt_tp, pl.datatypes.Date):
        return lambda dt: datetime.combine(dt, time())
    if isinstance(dt_tp, pl.datatypes.Datetime):
        return lambda dt: dt
    raise RuntimeError(f"Unable to convert {dt_tp} to a date or datetime")


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


class PolarsDataFrameSource(DataFrameSource):
    """
    A simple data frame source
    """

    def __init__(self, df: DATA_FRAME_SOURCE):
        self._df: DATA_FRAME_SOURCE = df

    @property
    def data_frame(self) -> DATA_FRAME_SOURCE:
        return self._df
