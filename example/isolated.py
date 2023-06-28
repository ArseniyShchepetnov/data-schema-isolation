"""Isolated data schema."""
import functools
from collections.abc import Callable

import pandas as pd
import pandera as pa
from pydantic import BaseModel


class Columns(BaseModel):
    """Data schema for columns."""

    id_: str = "Id"
    city: str = "City"
    height: str = "Height"
    name: str = "Name"

    class Config:
        """Pydantic configuration."""

        frozen = True


@functools.cache
def get_data_schema(columns: Columns) -> pa.DataFrameSchema:
    """Return data schema for a DataFrame."""
    return pa.DataFrameSchema(
        {
            columns.id_: pa.Column(
                pa.Int,
                checks=pa.Check.greater_than_or_equal_to(0),
                nullable=False,
            ),
            columns.city: pa.Column(pa.String, nullable=True),
            columns.height: pa.Column(
                pa.Float, checks=pa.Check.in_range(0, 3), nullable=False
            ),
            columns.name: pa.Column(
                pa.String,
                checks=pa.Check.str_matches("[A-Z][a-z]+"),
                nullable=False,
            ),
        }
    )


class DataIsolatedSchema:
    """Data wrapper for a DataFrame."""

    def __init__(self, data: pd.DataFrame, columns: Columns | None = None):
        """Initialize data wrapper."""
        if columns is None:
            columns = Columns()
        self.columns = columns
        schema = get_data_schema(columns)
        schema.validate(data)
        self.data = data

    @property
    def c(self) -> Columns:
        """Return column name."""
        return self.columns

    @property
    def n_samples(self) -> int:
        """Return number of samples."""
        return self.data.shape[0]

    @classmethod
    def from_csv(
        cls,
        path: str,
        preprocess: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        columns: Columns | None = None,
        **kwargs
    ) -> "DataIsolatedSchema":
        """Create data from a CSV file."""
        if columns is None:
            columns = Columns()

        data = pd.read_csv(path, **kwargs)
        if preprocess is not None:
            data = preprocess(pd.read_csv(path, **kwargs))

        return cls(data, columns=columns)

    def get_city_data(self, city: str) -> "DataIsolatedSchema":
        """Slice data for a city and preserve schema."""
        return DataIsolatedSchema(
            self.data[self.data[self.c.city] == city], columns=self.columns
        )

    def names(self) -> list[str]:
        """Return list of names."""
        return self.data[self.c.name].unique().tolist()
