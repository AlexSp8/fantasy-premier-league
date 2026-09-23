
import pandas as pd


def get_outlier_rows(df: pd.DataFrame, columns: list[str], iqr: float = 1.5) -> pd.DataFrame:
    """Get the rows in a pd.DataFrame with outliers in at least one of the specified columns.
    The outliers are calculated based on the IQR rule.
    Args:
        df: The pd.DataFrame of the data to be checked.
        columns: The list of columns to be examined.
        iqr (optional): The value of the IQR multiplier.
    Returns:
        A pd.DataFrame of the rows with at least one outlier."""

    df_stats = df[columns].describe()

    upper = df_stats.loc['75%'] + iqr*(df_stats.loc['75%'] - df_stats.loc['25%'])
    lower = df_stats.loc['25%'] - iqr*(df_stats.loc['75%'] - df_stats.loc['25%'])

    outliers = df.copy()

    outlier_cols = [c+'_is_outlier' for c in columns]
    for c in columns:
        outliers[c+'_is_outlier'] = (outliers[c] > upper[c]) | (outliers[c] < lower[c])

    mask = outliers[outlier_cols].any(axis=1)
    outliers = outliers[mask]

    return outliers

def get_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute the correlation matrix for a list of columns of a pd.DataFrame.
    Args:
        df: The pd.DataFrame to be checked.
        columns: The list of columns to be examined.
    Returns:
        A pd.DataFrame of the correlation matrix of the columns of the pd.DataFrame."""

    cor_matrix = df[columns].corr()

    return cor_matrix
