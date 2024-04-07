from datetime import date, datetime

import polars as pl
from hgraph import GraphConfiguration, evaluate_graph, graph, TSB, ts_schema, TS

from hgraph_db.data_source import PolarsDataFrameSource, SqlDataFrameSource, DataConnectionStore, \
    DataStore
from hgraph_db.data_source_generators import tsb_from_data_source


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
        return tsb_from_data_source(MockDataSource, "date")

    with DataStore() as store:
        config = GraphConfiguration()
        result = evaluate_graph(main, config)

    assert result == [
        (datetime(2020, 1, 1), {'name': 'John', 'age': 25}),
        (datetime(2020, 1, 2), {'name': 'Alice', 'age': 30}),
        (datetime(2020, 1, 3), {'name': 'Bob', 'age': 35})
    ]


CREATE_TBL_SQL = '''
CREATE TABLE my_table (
    date DATE,
    name TEXT,
    age INTEGER,
    PRIMARY KEY (date, name)
);
'''

INSERT_TEST_DATA = [
    "INSERT INTO my_table (date, name, age) VALUES ('2020-01-01', 'John', 25);",
    "INSERT INTO my_table (date, name, age) VALUES ('2020-01-02', 'Alice', 30);",
    "INSERT INTO my_table (date, name, age) VALUES ('2020-01-03', 'Bob', 35);"
]


def test_db_source():
    import duckdb
    conn = duckdb.connect(":memory:")
    conn.execute(CREATE_TBL_SQL)
    conn.commit()
    for ins in INSERT_TEST_DATA:
        conn.execute(ins)
    conn.commit()

    class AgeDataSource(SqlDataFrameSource):

        def __init__(self):
            super().__init__("SELECT date, name, age FROM my_table", "duckdb")

    @graph
    def main() -> TSB[ts_schema(name=TS[str], age=TS[int])]:
        return tsb_from_data_source(AgeDataSource, "date")

    with DataConnectionStore() as dsc, DataStore():
        dsc.set_connection("duckdb", conn)
        config = GraphConfiguration()
        result = evaluate_graph(main, config)

    assert result == [
        (datetime(2020, 1, 1), {'name': 'John', 'age': 25}),
        (datetime(2020, 1, 2), {'name': 'Alice', 'age': 30}),
        (datetime(2020, 1, 3), {'name': 'Bob', 'age': 35})
    ]

