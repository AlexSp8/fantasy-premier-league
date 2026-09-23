
import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator


class Bounds(BaseModel):
    """Class to define numerical boundaries for a variable."""
    lower: float
    upper: float

    @model_validator(mode='after')
    def check_bounds(self) -> Bounds:
        """Check if the upper field is correctly larger than the lower field.
        Runs after validating lower, upper individually.
        Raises:
            ValueError: If self.lower > self.upper."""
        if self.lower > self.upper:
            raise ValueError(f"lower bound {self.lower} cannot exceed upper bound {self.upper}")
        return self


class DataFeatures:
    """Class to add general data features in a pd.DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a DataFeatures instance with the pd.DataFrame to add features on.
        Args:
            df: The pd.DataFrame to add features on."""

        self.df = df.copy()

    def _columns_in_df(self, columns: list[str]) -> None:
        """Check if all specified columns are present in the pd.DataFrame.
        Args:
            columns: The list of columns names to check.
        Raises:
            KeyError: If any columns are missing from self.pd.DataFrame."""

        missing = set(columns) - set(self.df.columns)
        if missing:
            raise KeyError(f"The following columns were not found in the DataFrame: {sorted(missing)}")

    def calculate_per_x(self, columns: list[str], x_col: str, x: float) -> None:
        """Calculate per_x normalized values for the specified columns of the pd.DataFrame.
        Args:
            columns: The list of column names whose values to modify.
            x_col: The column name of the normalization variable.
            x: The normalization value."""

        self._columns_in_df(columns=[x_col])
        self._columns_in_df(columns=columns)

        suffix = '_per_' + str(int(x))
        for c in columns:
            new_c = c + suffix
            self.df[new_c] = ((x*self.df[c]/self.df[x_col])
                              .replace([np.inf, -np.inf], 0.0).fillna(value=0.0))

        return None

    def calculate_from_per_x(self, columns: list[str], x_col: str, x: float) -> None:
        """Calculate values from per_x normalized values for the specified columns of the pd.DataFrame.
        Args:
            columns: The list of column names to use for the calculations.
            x_col: The column name of the normalization variable.
            x: The normalization value."""

        self._columns_in_df(columns=[x_col])
        self._columns_in_df(columns=columns)

        suffix = '_per_' + str(int(x))
        for c in columns:
            new_c = c.removesuffix(suffix)
            self.df[new_c] = ((self.df[c]*self.df[x_col]/x)
                              .replace([np.inf, -np.inf], 0.0).fillna(value=0.0))

        return None

    def set_features_bounds(self, bounds_dict: dict[str, Bounds]) -> None:
        """Set numerical bounds on the specified columns of the pd.DataFrame.
        Args:
            bounds_dict: The dictionary mapping column names with lower and upper bounds."""

        for c, bounds in bounds_dict.items():
            l, u = bounds.lower, bounds.upper
            self.df[c] = self.df[c].clip(lower=l, upper=u)

        return None
