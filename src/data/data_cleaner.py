
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd


class DataCleaner:
    """Class to perform general data cleaning in a pd.DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a DataCleaner instance with the pd.DataFrame to be cleaned.
        Args:
            df: The pd.DataFrame to be cleaned."""

        self.df = df.copy()

    def clean_data(self) -> None:
        """Perform general cleaning on the pd.DataFrame."""

        self.set_numerical_columns()

        self.convert_to_hashable()

        self.df.drop_duplicates(inplace=True)

        return None

    def set_numerical_columns(self) -> None:
        """Convert ambiguous (object, str) columns of the pd.DataFrame (df)
        into numerical columns (float64 or int64). Fill invalid data with NaN."""

        ambiguous_cols = self.df.select_dtypes(include=['object', 'str']).columns.tolist()

        for col in ambiguous_cols:

            num_col = pd.to_numeric(arg=self.df[col], errors='coerce')

            col_nan = self.df[col].isnull().sum()
            num_nan = num_col.isnull().sum()

            if num_nan == col_nan:
                self.df[col] = num_col

        return None

    def convert_to_hashable(self) -> None:
        """Convert non-hashable columns of the pd.DataFrame into hashable tuples."""

        unhashable_types = (list, set, dict, np.ndarray)

        for col in self.df.columns:
            is_unhashable = self.df[col].apply(lambda x: isinstance(x, unhashable_types))
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

    def raise_negative_values(self, excluded_keywords: list[str] | None = None) -> None:
        """Check and raise error for negative values of numerical columns of the pd.DataFrame.
        Args:
            excluded_keywords: The list of column names keywords to not examine.
        Raises:
            ValueError: If incorrect negative values appear in the pd.DataFrame."""

        df_negative = self.get_negative_values(excluded_keywords=excluded_keywords)
        if not df_negative.empty:
            raise ValueError("DataFrame contains incorrect negative values!")

    def get_negative_values(self, excluded_keywords: list[str] | None = None) -> pd.DataFrame:
        """Get negative values of numerical columns of the pd.DataFrame.
        Args:
            excluded_keywords: The list of column names keywords to not examine.
        Returns:
            A pd.DataFrame containing the problematic rows of df."""

        if excluded_keywords is None:
            excluded_keywords = []

        num_cols = self.df.select_dtypes(include='number').columns.tolist()

        # set to avoid duplicates
        indices = set()

        for col in num_cols:
            negatives = self.df[col] < 0
            if negatives.sum() > 0 and not any(word in col for word in excluded_keywords):
                indices = indices.union(self.df[negatives].index)

        return self.df.loc[list(indices)]
