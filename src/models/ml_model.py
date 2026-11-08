
from typing import Any

from numpy.typing import ArrayLike
import pandas as pd
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


class MLModel:
    """Class with general Machine Learning models functionality."""

    def __init__(self, df: pd.DataFrame, features: list[str], target: str) -> None:
        """Initialize a MLModel instance with the pd.DataFrame and
        features + target columns to work with.
        Args:
            df: The pd.DataFrame to add features on.
            features: The list of column names of the pd.DataFrame acting as features.
            target: The column name of the pd.DataFrame acting as target variable."""

        self.df = df.copy()
        self.features = features.copy()
        self.target = target

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.scaler = None

    def _check_is_not_none(self, *args: Any, msg: str | None = None) -> None:
        """Ensure all provided arguments are not None.
        Args:
            *args: Variable number of attributes/arguments to check.
            msg (optional): Custom error message. If None, a default message is used.
        Raises:
            RuntimeError: If any argument in *args is None."""

        default_msg = "A train/test split and model training method must be called before this action."
        if any(arg is None for arg in args):
            raise RuntimeError(msg or default_msg)

    def split_historical_data(self, test_split: tuple[str, Any] | None = None) -> None:
        """Split historical data of thr pd.DataFrame in test/train
        on a specified historical column's value.
        Args:
            test_split (optional): The tuple of a column name and its value for the test/train split.
                None means all data is used for training."""

        if test_split is not None:
            col, val = test_split
            is_test = (self.df[col] == val)
        else:
            is_test = pd.Series(False, index=self.df.index)

        self.X_train = self.df.loc[~is_test, self.features]
        self.X_test = self.df.loc[is_test, self.features]

        self.y_train = self.df.loc[~is_test, self.target]
        self.y_test = self.df.loc[is_test, self.target]

    def get_model_errors(self, y_pred: ArrayLike) -> dict[str, float]:
        """Calculate model prediction metrics.
        Args:
            y_pred: The ArrayLike of the predicted values of the target.
        Returns:
            A dictionary of model prediction metrics."""

        self._check_is_not_none(self.y_test)

        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        rmse = metrics.root_mean_squared_error(self.y_test, y_pred)
        r2 = metrics.r2_score(self.y_test, y_pred)
        # mape = metrics.mean_absolute_percentage_error(self.y_test, y_pred)
        med_ae = metrics.median_absolute_error(self.y_test, y_pred)
        max_err = metrics.max_error(self.y_test, y_pred)

        predictions = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            # 'mape': mape,
            'median_ae': med_ae,
            'max_error': max_err
        }

        return predictions

    def scale_data(self) -> None:
        """Scale the features columns (dependent variables) of the train/test data."""

        self._check_is_not_none(self.X_train)

        self.X_train[self.features] = self.X_train[self.features].fillna(self.X_train.median())
        self.X_test[self.features] = self.X_test[self.features].fillna(self.X_train.median())

        self.scaler = StandardScaler()

        self.scaler.fit(X=self.X_train[self.features])

        self.X_train[self.features] = self.scaler.transform(X=self.X_train[self.features])
        self.X_test[self.features] = self.scaler.transform(X=self.X_test[self.features])


class XGBModel(MLModel):
    """Class to train and use an XGBoost Regression model."""

    def __init__(self, df: pd.DataFrame, features: list[str], target: str) -> None:
        """Initialize an XGBModel instance with the pd.DataFrame and
        features + target columns to work with.
        Args:
            df: The pd.DataFrame to add features on.
            features: The list of column names of the pd.DataFrame acting as features.
            target: The column name of the pd.DataFrame acting as target variable."""

        super().__init__(df, features, target)

        self.model = None

    def set_xgboost_model(
        self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3
        ) -> None:
        """Train the XGBoost model with the specified parameters.
        Args:
            n_estimators (optional): The number trees of the XGBoost model.
            max_depth (optional): The depth of each tree of the XGBoost model.
            learning_rate (optional): The learning rate of the XGBoost model."""

        self._check_is_not_none(self.X_train, self.y_train)

        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, learning_rate=learning_rate
            )
        self.model.fit(self.X_train, self.y_train)

    def get_features_importances(self) -> pd.Series:
        """Get the features' importances for the XGBoost model.
        Returns:
            A pd.Series of features' importances for the XGBoost model."""

        self._check_is_not_none(self.model)

        importances = pd.Series(self.model.feature_importances_, index=self.features)

        return importances
