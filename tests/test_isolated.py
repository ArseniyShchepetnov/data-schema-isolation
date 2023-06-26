"""Test isolated functions."""
import pandas as pd
import pandera as pa
import pytest

from example.isolated import Columns, DataIsolatedSchema


class Preprocess:
    """Test preprocess."""

    def __init__(self, columns: Columns):
        """Initialize preprocess."""
        self.columns = columns

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data."""
        data = data.astype(
            {
                self.columns.id_: "Int64",
                self.columns.height: "Float64",
                self.columns.name: "string",
                self.columns.city: "string",
            }
        )
        data.loc[data[self.columns.id_] == 3, self.columns.height] /= 1000
        data[self.columns.name] = (
            data[self.columns.name].str.strip().str.lower().str.title()
        )
        return data


def test_error_data() -> None:
    """Test error data."""
    with pytest.raises(pa.errors.SchemaError):
        DataIsolatedSchema.from_csv("tests/data.csv")


def test_preprocessed() -> None:
    """Test preprocessed data."""
    columns = Columns()
    data = DataIsolatedSchema.from_csv(
        "tests/data.csv", preprocess=Preprocess(columns)
    )
    assert data.n_samples == 5
    assert data.data[columns.height].max() < 3
    assert data.data[columns.name].str.istitle().sum() == 5
    assert data.data[columns.city].isna().sum() == 1
    assert data.data[columns.city].notna().sum() == 4


def test_get_city_data() -> None:
    """Test get city interface."""
    columns = Columns()
    data = DataIsolatedSchema.from_csv(
        "tests/data.csv", preprocess=Preprocess(columns)
    )
    paris_data = data.get_city_data("Paris")
    assert paris_data.n_samples == 2
    assert paris_data.data[columns.height].max() < 3
    assert paris_data.data[columns.name].str.istitle().sum() == 2
    assert paris_data.data[columns.city].isna().sum() == 0


def test_names() -> None:
    """Test get names interface."""
    columns = Columns()
    data = DataIsolatedSchema.from_csv(
        "tests/data.csv", preprocess=Preprocess(columns)
    )
    assert set(data.names()) == {"Alice", "Bob", "John"}
