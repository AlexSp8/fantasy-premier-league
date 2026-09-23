
from typing import Any

import pandas as pd

from config.settings import FPLSettings, ML_FEATURES_POS
from config.settings import POINTS_COEFFICIENTS, POINTS_COEFFICIENTS_POS
from src.data import data_store
from src.models.ml_model import MLModel, XGBModel


def set_predictions(
    df_features: pd.DataFrame, df_features_past: pd.DataFrame,
    fpl_settings: FPLSettings, store: bool = True
    ) -> pd.DataFrame:
    """Set players points predictions based on the analytical formula or a ML model.
    Args:
        df_features: The pd.DataFrame of live features.
        df_features_past: The pd.DataFrame of past features.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A pd.DataFrame of live features and players points predictions."""

    if fpl_settings.prediction_model == "formula":
        df_predict = set_fpl_formula_predictions(df_features=df_features)
    elif fpl_settings.prediction_model == "xgboost":
        target = fpl_settings.ml_target
        df_predict = set_fpl_ml_predictions(
            df_features=df_features, df_features_past=df_features_past, target=target
            )

    df_predict = df_predict.groupby('code', as_index=False).agg(
        player_name=('player_name', 'first'), position_name=('position_name', 'first'),
        team_name=('team_name', 'first'),
        now_cost=('now_cost', 'first'), xP_predicted_sum=('xP_predicted', 'sum'),
        # availability=('availability', 'first'),
        # xP_FPL_sum=('xP_FPL', 'sum'),
        )
    df_predict = df_predict.sort_values(by=['team_name', 'position_name'], ascending=[True, True])

    # print(df_predict.info())

    if store:
        file_path = "fpl_data/predict/players_data_xP.csv"
        data_store.save_df_to_csv(df=df_predict, file_path=file_path, overwrite=True)

    return df_predict

def set_fpl_formula_predictions(df_features: pd.DataFrame) -> pd.DataFrame:
    """Set players points predictions based on the analytical points formulas.
    Args:
        df_features: The pd.DataFrame of live features.
    Returns:
        A pd.DataFrame of players points predictions."""

    df = df_features.copy()
    df['xP_predicted'] = set_expected_points(df_features=df_features)

    return df

def set_expected_points(df_features: pd.DataFrame) -> pd.Series[float]:
    """Set the expected players points based on expected values of the related quantities.
    Args:
        df_features: The pd.DataFrame of live features.
    Returns:
        A pd.Series of the expected players points."""

    fdr_off = df_features['fdr_off']
    fdr_def = df_features['fdr_def']

    xMIN = df_features['MIN_form'] # minutes

    xG = df_features['xG_form']/fdr_off # goals
    xA = df_features['xA_form']/fdr_off # assists
    xGC = df_features['xGC_form']*fdr_def # goals conceded
    xSA = df_features['SA_form']*fdr_def # saves
    xCS = df_features['CS_form']/fdr_def # clean sheets
    xDC = df_features['DC_form']*fdr_def # defensive contributions
    xPS = df_features['PS_form'] # penalties saved
    xBP = df_features['BP_form'] # bonus points
    xYC = df_features['YC_form'] # yellow cards
    xRC = df_features['RC_form'] # red cards
    xOG = df_features['OG_form'] # own goals
    xPM = df_features['PM_form'] # penalties missed

    cxA = POINTS_COEFFICIENTS['xA']
    cxPS = POINTS_COEFFICIENTS['xPS']
    cxYC = POINTS_COEFFICIENTS['xYC']
    cxRC = POINTS_COEFFICIENTS['xRC']
    cxOG = POINTS_COEFFICIENTS['xOG']
    cxPM = POINTS_COEFFICIENTS['xPM']

    cxMIN_low = POINTS_COEFFICIENTS['xMIN_low']
    cxMIN_high = POINTS_COEFFICIENTS['xMIN_high']
    cxMIN_bar = POINTS_COEFFICIENTS['xMIN_bar']

    cxSA = POINTS_COEFFICIENTS_POS['GKP']['xSA']
    cxGC = POINTS_COEFFICIENTS['xGC']

    cxCS_gkp = POINTS_COEFFICIENTS_POS['GKP']['xCS']
    cxCS_def = POINTS_COEFFICIENTS_POS['DEF']['xCS']
    cxCS_mid = POINTS_COEFFICIENTS_POS['MID']['xCS']

    cxDC = POINTS_COEFFICIENTS['xDC']
    cxDC_bar_def = POINTS_COEFFICIENTS_POS['DEF']['xDC_bar']
    cxDC_bar_mid = POINTS_COEFFICIENTS_POS['MID']['xDC_bar']

    cxG_gkp = POINTS_COEFFICIENTS_POS['GKP']['xG']
    cxG_def = POINTS_COEFFICIENTS_POS['DEF']['xG']
    cxG_mid = POINTS_COEFFICIENTS_POS['MID']['xG']
    cxG_fwd = POINTS_COEFFICIENTS_POS['FWD']['xG']

    xC = cxA*xA + cxPS*xPS + xBP - cxYC*xYC - cxRC*xRC - cxOG*xOG - cxPM*xPM # common

    played = xMIN > 0
    xC[played] += cxMIN_low
    xMIN_more_than_60 = xMIN >= cxMIN_bar
    xC[xMIN_more_than_60] += (cxMIN_high-cxMIN_low)

    xP = pd.Series(index=df_features.index, dtype=float)

    gkp_mask = df_features["position_name"] == "GKP"
    xP[gkp_mask] = cxG_gkp*xG - cxGC*xGC + cxSA*xSA + cxCS_gkp*xCS + xC

    def_mask = df_features["position_name"] == "DEF"
    xP[def_mask] = cxG_def*xG - cxGC*xGC + cxCS_def*xCS + xC
    xDC10 = xDC >= cxDC_bar_def
    xP[def_mask & xDC10] += cxDC

    mid_mask = df_features["position_name"] == "MID"
    xP[mid_mask] = cxG_mid*xG + cxCS_mid*xCS + xC
    xDC12 = xDC >= cxDC_bar_mid
    xP[mid_mask & xDC12] += cxDC

    fwd_mask = df_features["position_name"] == "FWD"
    xP[fwd_mask] = cxG_fwd*xG + xC
    xP[fwd_mask & xDC12] += cxDC

    return xP

def set_fpl_ml_predictions(
    df_features: pd.DataFrame, df_features_past: pd.DataFrame, target: str
    ) -> pd.DataFrame:
    """Set points predictions for each player based on a ML model for each player position.
    Args:
        df_features: The pd.DataFrame of live features.
        df_features_past: The pd.DataFrame of past features.
        target: The target column for ML predictions.
    Returns:
        A pd.DataFrame of the ML points predictions for each player."""

    df = df_features.copy()
    df['xP_predicted'] = pd.Series(dtype=float)

    test_split = ('season', '2025-26')
    for pos, ft in ML_FEATURES_POS.items():

        df_pos = df_features_past[df_features_past['position_name'] == pos]

        # compare with baseline (target form)
        test_model = set_position_model(df_pos=df_pos, features=ft, target=target, test_split=test_split)
        base_model = set_base_model(df=df_pos, target=target, test_split=test_split)
        metrics = evaluate_position_model(test_model=test_model, base_model=base_model)

        print(f"\n{pos}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # all data for training
        xgb_model = set_position_model(df_pos=df_pos, features=ft, target=target, test_split=None)
        imp = xgb_model.get_features_importances()
        print(f"\n{pos} features importance\n{imp.sort_values(ascending=False)}")

        rows = (df['position_name'] == pos)
        if rows.any():
            df.loc[rows, 'xP_predicted'] = predict_position_points(df_features=df, xgb_model=xgb_model, rows=rows)

    return df

def set_position_model(
    df_pos: pd.DataFrame, features: list[str], target: str,
    test_split: tuple[str, Any] | None = None,
    ) -> XGBModel:
    """Set an XGBoost regression model for a player position.
    Args:
        df_pos: The pd.DataFrame of past player position data.
        features: The list of player position features.
        test_split (optional): The tuple (col, val) from which test data are chosen.
            None means all data is used for training.
    Returns:
        The fitted XGBoost model for a player position."""

    xgb_model = XGBModel(df=df_pos, features=features, target=target)

    xgb_model.split_historical_data(test_split=test_split)

    xgb_model.set_xgboost_model(n_estimators=100, max_depth=3, learning_rate=0.1)

    return xgb_model

def set_base_model(
    df: pd.DataFrame, target: str, test_split: tuple[str, Any] | None = None
    ) -> MLModel:
    """Set an XGBoost baseline model for a player position.
    Args:
        df: The pd.DataFrame of past player position data.
        test_split (optional): The tuple (col, val) from which test data are chosen.
            None means all data is used for training.
    Returns:
        The baseline XGBoost model for a player position."""

    base_model = MLModel(df=df, features=[target+'_form'], target=target)

    base_model.split_historical_data(test_split=test_split)

    return base_model

def evaluate_position_model(test_model: XGBModel, base_model: XGBModel) -> dict[str, dict[str, float]]:
    """Evaluate the XGBoost model (fitted and baseline) metrics.
    Args:
        test_model: The fitted XGBModel instance.
        test_model: The baseline XGBModel instance.
    Returns:
        The dictionary of fitted and baseline model metrics."""

    metrics: dict[str, dict[str, float]] = {}

    col = base_model.features
    y_pred = base_model.X_test[col]
    metrics['base'] = base_model.get_model_errors(y_pred=y_pred)

    y_pred = test_model.model.predict(test_model.X_test)
    metrics['test'] = test_model.get_model_errors(y_pred=y_pred)

    return metrics

def predict_position_points(
    df_features: pd.DataFrame, xgb_model: XGBModel, rows: pd.Series[bool],
    ) -> pd.Series[float]:
    """Predict player points using the XGBModel for a player position.
    Args:
        df_features: The pd.DataFrame of live features. Used for consistent indexing.
        xgb_model: The fitted XGBModel for a player position.
        rows: Boolean Series indicating rows of the player position.
    Returns:
        The pd.Series of the points predictions for the target rows."""

    X_test = df_features.loc[rows, xgb_model.features]
    return xgb_model.model.predict(X_test)
