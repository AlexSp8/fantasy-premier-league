
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from config.settings import ML_FEATURES_DICT


def scale_seasons_data(df_past: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Scale the data of th specified columns of a DataFrame (df).
    Args:
        df_past: The pd.DataFrame containing all past seasons data.
        columns: The list of columns of df to be scaled.
    Returns:
        The pd.DataFrame with scaled columns."""

    df_scaled = df_past.copy()
    df_scaled[columns] = df_scaled[columns].fillna(df_scaled[columns].median())

    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(X=df_scaled[columns])

    return df_scaled

def split_seasons_data(
        df_seasons_merged: pd.DataFrame, features_columns: list[str], target_column: str,
        test_season: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split past seasons data in test/train on a specified column.
    Args:
        df_seasons_merged: The pd.DataFrame containing all past seasons data.
        features_column: The features column names.
        target_column: The target column name.
        test_season (optional): The FPL season from which test data are chosen.
            Defaults to None (using all data for training).
    Returns:
        The tuple of independent (DataFrame) and dependent (Series) variable of test/train."""

    is_test = (
        df_seasons_merged['season'] == test_season
        if test_season is not None
        else pd.Series(False, index=df_seasons_merged.index)
    )

    X_train = df_seasons_merged.loc[~is_test, features_columns]
    X_test = df_seasons_merged.loc[is_test, features_columns]

    y_train = df_seasons_merged.loc[~is_test, target_column]
    y_test = df_seasons_merged.loc[is_test, target_column]

    return X_train, X_test, y_train, y_test

def evaluate_model_predictions(
        y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calculates the MAE and RMSE as model prediction metrics.
    Args:
        y_true: The ArrayLike representing the true values of the target.
        y_pred: TThe ArrayLike representing the predicted values of the target.
    Returns:
        A dictionary containing the MAE and RMSE for a model's predictions."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    predictions: dict[str, float] = {'mae': mae, 'rmse': rmse}

    return predictions

def set_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series[float],
    n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1) -> XGBRegressor:
    """Train an XGBoost regression model on the specified training data.
    Args:
        X_train: The pd.DataFrame containing the training features data.
        y_train: The pd.Series containing the training target data.
        n_estimators: The number trees of the XGBoost model.
        max_depth: The depth of each tree of the XGBoost model.
        learning_rate: The learning rate of the XGBoost model.
    Returns:
        The fitted XGBoost model object."""

    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         random_state=42, learning_rate=learning_rate)
    model.fit(X_train, y_train)

    return model

def set_position_models(df_seasons_merged: pd.DataFrame,
    test_season: str | None = None) -> dict[str, XGBRegressor]:
    """Set an XGBoost regression model each player position.
    Args:
        df_seasons_merged: The pd.DataFrame containing all past seasons data.
        test_season (optional): The FPL season from which test data are chosen.
            Defaults to None (using all data for training).
    Returns:
        The dictionary of fitted XGBoost model objects."""

    models_dict: dict[str, XGBRegressor] = {}

    for pos, ft in ML_FEATURES_DICT.items():

        df: pd.DataFrame = df_seasons_merged[df_seasons_merged['position_name'] == pos]

        X_train, _, y_train, _ = split_seasons_data(
            df_seasons_merged=df, features_columns=ft, target_column='PTS_GW', test_season=test_season)

        models_dict[pos] = set_xgboost_model(X_train=X_train, y_train=y_train,
                                            n_estimators=100, max_depth=3, learning_rate=0.1)

    return models_dict

def get_features_importances(models_dict: dict[str, XGBRegressor]
                             ) -> dict[str, pd.Series[float]]:
    """Get the feature importances for each ML model.
    Args:
        models_dict: The dictionary of fitted XGBoost model objects.
    Returns:
        The dictionary of pd.Series of features importances for each ML model."""

    importances: dict[str, pd.Series[float]] = {}
    for pos, ft in ML_FEATURES_DICT.items():
        importances[pos] = pd.Series(models_dict[pos].feature_importances_, index=ft)

    return importances

def evaluate_position_models(df_seasons_merged: pd.DataFrame,
    models_dict: dict[str, XGBRegressor], test_season: str) -> dict[str, dict[str, float]]:
    """Evaluate the XGBoost regression model metrics for each player position
    along with the baseline metrics.
    Args:
        df_seasons_merged: The pd.DataFrame containing all past seasons data.
        models_dict: The dictionary of fitted XGBoost model objects.
        test_season: The FPL season from which test data are chosen.
    Returns:
        The dictionary of baseline and model prediction metrics for each position."""

    metrics_dict: dict[str, dict[str, float]] = {}

    for pos, ft in ML_FEATURES_DICT.items():

        df: pd.DataFrame = df_seasons_merged[df_seasons_merged['position_name'] == pos]

        _, X_test_base, _, y_test_base = split_seasons_data(
            df_seasons_merged=df, features_columns=['xPTS_GW'], target_column='PTS_GW', test_season=test_season)

        metrics_dict[pos+'_base'] = evaluate_model_predictions(y_true=y_test_base, y_pred=X_test_base['xPTS_GW'])

        _, X_test, _, y_test = split_seasons_data(
            df_seasons_merged=df, features_columns=ft, target_column='PTS_GW', test_season=test_season)

        y_pred = models_dict[pos].predict(X_test)

        metrics_dict[pos] = evaluate_model_predictions(y_true=y_test, y_pred=y_pred)

    return metrics_dict

def set_fpl_ml_predictions(df_players_features: pd.DataFrame,
                        models_dict: dict[str, XGBRegressor]) -> pd.DataFrame:
    """Set FPL predictions for each player based on the ML model for each position.
    Args:
        df_players_features: The pd.DataFrame containing the FPL features data.
        models_dict: The dictionary of fitted XGBoost model objects.
    Returns:
        A pd.DataFrame containing the ML predictions for FPL points for each player."""

    df = df_players_features.copy()

    df['xP_predicted'] = predict_expected_points(df_players_features=df, models_dict=models_dict)

    df = df.groupby('code', as_index=False).agg(
        player_name=('player_name', 'first'), position_name=('position_name', 'first'),
        team_name=('team_name', 'first'), availability=('availability', 'first'),
        now_cost=('now_cost', 'first'), xP_predicted_sum=('xP_predicted', 'sum'),
        # xP_FPL_sum=('xP_FPL', 'sum'),
        )
    df = df.sort_values(by=['team_name', 'position_name'], ascending=[True, True])

    return df.copy()

def predict_expected_points(df_players_features: pd.DataFrame,
                            models_dict: dict[str, XGBRegressor]) -> pd.Series[float]:
    """Predict player points using the ML model for each player position.
    Args:
        df_players_features: The pd.DataFrame containing current season data.
        models_dict: The dictionary of fitted XGBoost model objects.
    Returns:
        The pd.Series containing the 'xP_predicted' column."""

    xP: pd.Series[float] = pd.Series(index=df_players_features.index, dtype=float)

    for pos, ft in ML_FEATURES_DICT.items():

        rows: pd.Series[bool] = (df_players_features['position_name'] == pos)

        X_test: pd.DataFrame = df_players_features.loc[rows, ft].copy()

        xP[rows] = models_dict[pos].predict(X_test)

    return xP
