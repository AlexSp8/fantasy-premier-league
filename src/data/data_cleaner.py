
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd


class DataCleaner:

    def __init__(self, df: pd.DataFrame):

        self.df: pd.DataFrame = df.copy()

    def clean_data(self) -> None:
        """Check and clean data columns of a pd.DataFrame (df)."""

        self.set_numerical_columns()

        self.convert_to_hashable()

        self.df.drop_duplicates(inplace=True)

        return None

    def set_numerical_columns(self) -> None:
        """Convert ambiguous (object, str) columns of a pd.DataFrame (df)
        into numerical columns (float64 or int64). Fill invalid data with NaN."""

        ambiguous_cols: list[str] = self.df.select_dtypes(include=['object', 'str']).columns.tolist()

        for col in ambiguous_cols:

            num_col: pd.Series[np.float64 | np.int64] = pd.to_numeric(arg=self.df[col], errors='coerce')

            col_nan: int = self.df[col].isnull().sum()
            num_nan: int = num_col.isnull().sum()

            if num_nan == col_nan:
                self.df[col] = num_col

        return None

    def convert_to_hashable(self) -> None:
        """Convert non-hashable columns of a pd.DataFrame (df) into hashable tuples."""

        unhashable_types: tuple[type, ...] = (list, set, dict, np.ndarray)

        for col in self.df.columns:
            is_unhashable: pd.Series[bool] = self.df[col].apply(lambda x: isinstance(x, unhashable_types))
            if is_unhashable.any():
                self.df[col] = self.df[col].apply(self.make_hashable)

        return None

    def make_hashable(self, x: Any) -> Any:
        """Recursively convert unhashable Python objects into hashable ones.
        - Lists, sets, and NumPy arrays are converted into tuples.
        - Dictionaries and Mappings are converted into sorted tuples of (key, value) pairs.
        - Primitive hashable types (str, int, float, bytes, tuples) are returned unchanged.
        Args:
            x: Any Python object to evaluate and convert.
        Returns:
            A hashable equivalent of the input object (or the original value if hashable)."""

        if isinstance(x, (str, bytes)):
            return x
        elif isinstance(x, Mapping):
            return tuple(sorted((k, self.make_hashable(v)) for k, v in x.items()))
        elif isinstance(x, (list, set, np.ndarray)):
            return tuple(self.make_hashable(item) for item in x)
        return x

    def flag_negative_values(self, excluded_keywords: list[str] = None) -> pd.DataFrame:
        """Flag negative values of numerical columns of a pd.DataFrame (df).
        Args:
            excluded_keywords: The keywords in column names to not examined.
        Returns:
            A pd.DataFrame containing the problematic rows of df."""

        if excluded_keywords is None:
            excluded_keywords = []

        num_cols: list[str] = self.df.select_dtypes(include='number').columns.tolist()

        indices: set = set()

        for col in num_cols:
            negatives: pd.Series[bool] = self.df[col] < 0
            if negatives.sum() > 0 and not any(word in col for word in excluded_keywords):
                indices = indices.union(self.df[negatives].index)

        return self.df.loc[list(indices)]
