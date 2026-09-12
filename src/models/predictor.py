
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class ModelPredictor:

    def __init__(self, df: pd.DataFrame, features_columns: list[str], target_column: str):
        """
        Args:
            df: The pd.DataFrame containing the data.
            features_columns: The columns of df which will be used as feature for the training.
            target_column: The column name of the target (dependent) variable."""

        self.df = df.copy()
        self.features_columns = features_columns
        self.target_column = target_column

    def scale_data(self) -> pd.DataFrame:
        """Scale the data of th specified columns of a DataFrame (df).
        Args:
            df: The pd.DataFrame.
            columns: The list of columns of df to be scaled.
        Returns:
            The pd.DataFrame with scaled columns."""

        columns = self.features_columns
        df_scaled = self.df.copy()
        df_scaled[columns] = df_scaled[columns].fillna(df_scaled[columns].median())

        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(X=df_scaled[columns])

        return df_scaled

    def check_outliers(self, columns: list[str], iqr_value: float = 1.5) -> pd.DataFrame:
        """Returns a pd.DataFrame of boolean values for outliers in the specified columns
        of a pd.DataFrame (df). The outliers are calculated based on the IQR rule.
        Args:
            df: The pd.DataFrame to be studied.
            columns: The list of columns to be examined.
            iqr_value (optional): The value of the IQR multiplier.
        Returns:
            A pd.DataFrame with boolean flags indicating outliers in the specified columns of df."""

        df_stats: pd.DataFrame = self.df[columns].describe()

        upper_bound: pd.DataFrame = df_stats.loc['75%'] + iqr_value*(df_stats.loc['75%'] - df_stats.loc['25%'])
        lower_bound: pd.DataFrame = df_stats.loc['25%'] - iqr_value*(df_stats.loc['75%'] - df_stats.loc['25%'])

        df_out: pd.DataFrame = self.df.copy()

        for c in columns:
            df_out[c+'_is_outlier'] = (df_out[c] > upper_bound[c]) | (df_out[c] < lower_bound[c])

        # seaborn.boxplot(data=df_out, y='total_points')
        # plt.show()

        # By position
        # group = df_out.groupby('position_name').agg({'total_points': 'mean'})
        # seaborn.boxplot(data=df_out, x='position_name', y='total_points')
        # plt.show()
        # print(group.sort_values('total_points', ascending=False))

        return df_out

    def check_multi_co_linearity(self) -> None:
        """Computes the correlation matrix and displays the corresponding heatmap.
        Args:
            columns: The list of columns of df to be checked.
        """

        columns = self.features_columns
        cor_matrix: pd.DataFrame = self.df[columns].corr()
        # print(cor_matrix)

        # seaborn.heatmap(data=cor_matrix, annot=True)
        # plt.show()

    def split_data(self, stratify_column: str,
                   test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits the data in test/train stratifying on a specified column.
        Args:
            stratify_column: The column name of the data upon which stratify is applied.
        Returns:
            The tuple of independent (DataFrame) and dependent (Series) variable of test/train."""

        X_features: pd.DataFrame = self.df[self.features_columns]
        y_target: pd.Series = self.df[self.target_column]

        stratify_array: pd.Series = self.df[stratify_column]

        X_train, X_test, y_train, y_test = train_test_split(X_features, y_target,
            test_size=test_size, random_state=42, stratify=stratify_array)

        return X_train, X_test, y_train, y_test

    def set_train_test_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series[Any], y_test: pd.Series[Any]):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def baseline_mae_model(self) -> float:
        """Calculates the MAE as a baseline model prediction.
        Args:
            y_train: The train dataset.
            y_test: The test dataset.
        Returns:
            The Mean Absolute Error between y_test and the predictions."""

        y_train_mean: float = self.y_train.mean()

        y_pred = np.full(shape=self.y_test.shape, fill_value=y_train_mean)

        return mean_absolute_error(self.y_test, y_pred)

    def train_xgboost_model(self) -> XGBRegressor:
        """Trains an XGBoost regression model.
        Args:
            X_train: The training features.
            y_train: The training target.
        Returns:
            The fitted XGBoost model."""

        model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(self.X_train, self.y_train)

        return model

    def set_model(self, model: XGBRegressor):
        self.model = model

    def xgboost_mea(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculates the MAE for the XGBoost predictions.
        Args:
            model: The XGBoost model instance.
            X: The test dataset.
            y: The test dataset.
        Returns:
            The Mean Absolute Error between y and the predictions."""

        y_pred = self.model.predict(X)

        return mean_absolute_error(y, y_pred)
