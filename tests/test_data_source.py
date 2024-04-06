from datetime import date, datetime

import polars as pl
from hgraph import GraphConfiguration, evaluate_graph, graph, TSB, ts_schema, TS

from hgraph_db.data_source import PolarsDataFrameSource, data_frame_source


class MockDataSource(PolarsDataFrameSource):

    def __init__(self):
        df = pl.DataFrame({
            'date': [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            'name': ['John', 'Alice', 'Bob'],
            'age': [25, 30, 35]
        })
        super().__init__(df)


def test_data_source():
    @graph
    def main() -> TSB[ts_schema(name=TS[str], age=TS[int])]:
        return data_frame_source(MockDataSource, "date")

    config = GraphConfiguration()
    result = evaluate_graph(main, config)
    assert result == [
        (datetime(2020, 1, 1), {'name': 'John', 'age': 25}),
        (datetime(2020, 1, 2), {'name': 'Alice', 'age': 30}),
        (datetime(2020, 1, 3), {'name': 'Bob', 'age': 35})
    ]
