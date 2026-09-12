
import pandas as pd

class DataFeatures:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

    def columns_in_df(self, columns: list[str]) -> None:
        """Check if all specified columns are present in a pd.DataFrame.
        Args:
            columns: The columns names to check."""

        for c in columns:
            if c not in self.df.columns:
                raise KeyError(f"Column {c} not in pd.DataFrame!")

    def replace_columns(self, columns: list[str], new_columns: list[str]) -> None:
        """Replace the specified columns of a pd.DataFrame with the
        specified new columns of the same pd.DataFrame.
        Args:
            columns: The list of column names to be replaced.
            new_columns: The list of column names to replace with."""

        self.df[columns] = self.df[new_columns]

        self.df = self.df.drop(columns=new_columns)

        rename_dict: dict[str, str] = dict(zip(columns, new_columns))
        self.df = self.df.rename(columns=rename_dict)

        return None

    def calculate_expected_from_total(self, columns: list[str], n: int) -> None:
        """Calculate the expected values as the mean from total stats.
        Args:
            columns: The list of column names containing total stats.
            n: The number of data comprising the total stat."""

        self.columns_in_df(columns=columns)

        n: int = max(n, 1)

        for c in columns:
            self.df['x'+c] = (self.df[c]/n).clip(lower=0.0).fillna(0.0)

        return None

    def convert_to_expected_from_total(self, columns: list[str], n: int) -> None:
        """Convert (replace) the total values to expected (as mean) from total stats.
        Args:
            columns: The list of column names to perform the calculation.
            n: The number of data comprising the total stat."""

        self.calculate_expected_from_total(columns=columns, n=n)

        new_columns: list[str] = ['x'+c for c in columns]

        self.replace_columns(columns=columns, new_columns=new_columns)

        return None

    def calculate_per_x(self, columns: list[str], x_col: str, x: float = 90.0) -> None:
        """Calculate per_x normalized values for the specified columns
        of a pd.DataFrame.
        Args:
            columns: The list of column names whose values to modify.
            x_col: The column name of the normalization variable.
            x (optional): The normalization value. Defaults to 90."""

        self.columns_in_df(columns=[x_col])
        self.columns_in_df(columns=columns)

        for c in columns:
            self.df[c+'_per_'+str(int(x))] = (x*self.df[c]/self.df[x_col]).fillna(value=0.0)

        return None

    def convert_to_per_x(self, columns: list[str], x_col: str, x: float = 90.0) -> None:
        """Convert (replace) specified columns of a pd.DataFrame
        with the corresponding per_x normalized values.
        Args:
            columns: The list of column names whose values to modify.
            x_col: The column name of the normalization variable.
            x (optional): The normalization value. Defaults to 90."""

        self.calculate_per_x(columns=columns, x_col=x_col, x=x)

        new_columns: list[str] = [c+'_per_'+str(int(x)) for c in columns]

        self.replace_columns(columns=columns, new_columns=new_columns)

        return None

    def calculate_from_per_x(self, columns: list[str], x_col: str, x: float = 90.0) -> None:
        """Calculate values from per_x normalized values
        for the specified columns of a pd.DataFrame.
        Args:
            columns: The list of column names to use for the calculations.
            x_col: The column name of the normalization variable.
            x (optional): The normalization value. Defaults to 90."""

        self.columns_in_df(columns=[x_col])
        self.columns_in_df(columns=columns)

        for c in columns:
            suffix: str = '_per_'+str(int(x))
            new_c: str = c.removesuffix(suffix)
            self.df[new_c] = (self.df[c]*self.df[x_col]/x).fillna(value=0.0)

        return None

    def convert_from_per_x(self, columns: list[str],
                        x_col: str, x: float = 90.0) -> None:
        """Calculate values from per_x normalized values
        for the specified columns of a pd.DataFrame.
        Args:
            columns: The list of column names to use for the calculations.
            x_col: The column name of the normalization variable.
            x (optional): The normalization value. Defaults to 90."""

        self.calculate_from_per_x(columns=columns, x_col=x_col, x=x)

        new_columns: list[str] = []
        for c in columns:
            suffix: str = '_per_'+str(int(x))
            new_c: str = c.removesuffix(suffix)
            new_columns.append(new_c)

        self.replace_columns(columns=columns, new_columns=new_columns)

        return None

    def set_features_bounds(self, bounds_dict: dict[str, list[float]]) -> None:
        """Set numerical bounds on the specified columns of a pd.DataFrame.
        Args:
            bounds_dict: The dictionary mapping column names with lower and upper bounds."""

        for key, value in bounds_dict.items():
            c, u, l = key, value[0], value[1]
            self.df[c] = self.df[c].clip(lower=l, upper=u)

        return None
