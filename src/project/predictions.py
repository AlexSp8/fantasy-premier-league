
from typing import Any

import pandas as pd

from config.naming import Names
from config.settings import FPLSettings, Columns, ML_FEATURES_POS_FORM, PAST_SEASONS
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

    if fpl_settings.prediction_model == Names.FORMULA:
        df_predict = set_fpl_formula_predictions(df_features)
    elif fpl_settings.prediction_model == Names.XGBOOST:
        target = fpl_settings.ml_target
        df_predict = set_fpl_ml_predictions(df_features, df_features_past, target)

    df_predict = df_predict.groupby(Columns.PLAYER_CODE, as_index=False).agg(
        player_name=(Columns.PLAYER_NAME, 'first'), position_name=(Columns.POSITION_NAME, 'first'),
        team_name=(Columns.TEAM_NAME, 'first'), player_cost=(Columns.PLAYER_COST, 'first'),
        xPTS_predicted_sum=(Columns.EXPECTED_POINTS_PREDICTED, 'sum'),
        # availability=('availability', 'first'),
        # xP_FPL_sum=('xP_FPL', 'sum'),
        )
    df_predict = df_predict.sort_values(by=[Columns.TEAM_NAME, Columns.POSITION_NAME], ascending=[True, True])

    # print(df_predict.info())

    if store:
        file_path = "fpl_data/predict/players_data_xP.csv"
        data_store.save_df_to_csv(df_predict, file_path, overwrite=True)

    return df_predict

def set_fpl_formula_predictions(df_features: pd.DataFrame) -> pd.DataFrame:
    """Set players points predictions based on the analytical points formulas.
    Args:
        df_features: The pd.DataFrame of live features.
    Returns:
        A pd.DataFrame of players points predictions."""

    df = df_features.copy()
    df[Columns.EXPECTED_POINTS_PREDICTED] = set_expected_points(df_features)

    return df

def set_expected_points(df_features: pd.DataFrame) -> pd.Series:
    """Set the expected players points based on expected values of the related quantities.
    Args:
        df_features: The pd.DataFrame of live features.
    Returns:
        A pd.Series of the expected players points."""

    fdr_off = df_features[Columns.FDR_OFF]
    fdr_def = df_features[Columns.FDR_DEF]

    xMIN = df_features[Columns.MINUTES_FORM]

    xG = df_features[Columns.EXPECTED_GOALS_FORM]/fdr_off
    xA = df_features[Columns.EXPECTED_ASSISTS_FORM]/fdr_off
    xGC = df_features[Columns.EXPECTED_GOALS_CONCEDED_FORM]*fdr_def
    xSA = df_features[Columns.SAVES_FORM]*fdr_def
    xCS = df_features[Columns.CLEAN_SHEETS_FORM]/fdr_def
    xDC = df_features[Columns.DEFENSIVE_CONTRIBUTIONS_FORM]*fdr_def
    xPS = df_features[Columns.PENALTIES_SAVED_FORM]
    xBP = df_features[Columns.BONUS_POINTS_FORM]
    xYC = df_features[Columns.YELLOW_CARDS_FORM]
    xRC = df_features[Columns.RED_CARDS_FORM]
    xOG = df_features[Columns.OWN_GOALS_FORM]
    xPM = df_features[Columns.PENALTIES_MISSED_FORM]

    cxA = POINTS_COEFFICIENTS[Names.ASSIST]
    cxPS = POINTS_COEFFICIENTS[Columns.PENALTIES_SAVED]
    cxYC = POINTS_COEFFICIENTS[Names.YELLOW_CARD]
    cxRC = POINTS_COEFFICIENTS[Names.RED_CARD]
    cxOG = POINTS_COEFFICIENTS[Names.OWN_GOALS]
    cxPM = POINTS_COEFFICIENTS[Columns.PENALTIES_MISSED]

    cxMIN_low = POINTS_COEFFICIENTS['xMIN_low']
    cxMIN_high = POINTS_COEFFICIENTS['xMIN_high']
    cxMIN_bar = POINTS_COEFFICIENTS['xMIN_bar']

    cxSA = POINTS_COEFFICIENTS_POS[Names.GOALKEEPER][Names.SAVES]
    cxGC = POINTS_COEFFICIENTS[Names.GOAL_CONCEDED]

    cxCS_gkp = POINTS_COEFFICIENTS_POS[Names.GOALKEEPER][Names.CLEAN_SHEET]
    cxCS_def = POINTS_COEFFICIENTS_POS[Names.DEFENDER][Names.CLEAN_SHEET]
    cxCS_mid = POINTS_COEFFICIENTS_POS[Names.MIDFIELDER][Names.CLEAN_SHEET]

    cxDC = POINTS_COEFFICIENTS[Names.DEFENSIVE_CONTRIBUTIONS]
    cxDC_bar_def = POINTS_COEFFICIENTS_POS[Names.DEFENDER]['xDC_bar']
    cxDC_bar_mid = POINTS_COEFFICIENTS_POS[Names.MIDFIELDER]['xDC_bar']

    cxG_gkp = POINTS_COEFFICIENTS_POS[Names.GOALKEEPER][Names.GOAL]
    cxG_def = POINTS_COEFFICIENTS_POS[Names.DEFENDER][Names.GOAL]
    cxG_mid = POINTS_COEFFICIENTS_POS[Names.MIDFIELDER][Names.GOAL]
    cxG_fwd = POINTS_COEFFICIENTS_POS[Names.FORWARD][Names.GOAL]

    xC = cxA*xA + cxPS*xPS + xBP - cxYC*xYC - cxRC*xRC - cxOG*xOG - cxPM*xPM # common

    played = xMIN > 0
    xC[played] += cxMIN_low
    xMIN_more_than_60 = xMIN >= cxMIN_bar
    xC[xMIN_more_than_60] += (cxMIN_high-cxMIN_low)

    xP = pd.Series(index=df_features.index, dtype=float)

    gkp_mask = df_features[Columns.POSITION_NAME] == Names.GOALKEEPER
    xP[gkp_mask] = cxG_gkp*xG - cxGC*xGC + cxSA*xSA + cxCS_gkp*xCS + xC

    def_mask = df_features[Columns.POSITION_NAME] == Names.DEFENDER
    xP[def_mask] = cxG_def*xG - cxGC*xGC + cxCS_def*xCS + xC
    xDC10 = xDC >= cxDC_bar_def
    xP[def_mask & xDC10] += cxDC

    mid_mask = df_features[Columns.POSITION_NAME] == Names.MIDFIELDER
    xP[mid_mask] = cxG_mid*xG + cxCS_mid*xCS + xC
    xDC12 = xDC >= cxDC_bar_mid
    xP[mid_mask & xDC12] += cxDC

    fwd_mask = df_features[Columns.POSITION_NAME] == Names.FORWARD
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
    df[Columns.EXPECTED_POINTS_PREDICTED] = pd.Series(dtype=float)

    test_split = (Columns.SEASON, PAST_SEASONS[-1])
    for pos, ft in ML_FEATURES_POS_FORM.items():

        df_pos = df_features_past[df_features_past[Columns.POSITION_NAME] == pos]

        # compare with baseline (target form)
        test_model = set_position_model(df_pos, ft, target, test_split)
        base_model = set_base_model(df_pos, target, test_split)
        metrics = evaluate_position_model(test_model, base_model)

        print(f"\n{pos}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # all data for training
        xgb_model = set_position_model(df_pos, ft, target, test_split=None)
        imp = xgb_model.get_features_importances()
        print(f"\n{pos} features importance\n{imp.sort_values(ascending=False)}")

        rows = (df[Columns.POSITION_NAME] == pos)
        if rows.any():
            df.loc[rows, Columns.EXPECTED_POINTS_PREDICTED] = predict_position_points(df, xgb_model, rows)

    return df

def set_position_model(
    df_pos: pd.DataFrame, features: list[str], target: str, test_split: tuple[str, Any] | None = None,
    ) -> XGBModel:
    """Set an XGBoost regression model for a player position.
    Args:
        df_pos: The pd.DataFrame of past player position data.
        features: The list of player position features.
        test_split (optional): The tuple (col, val) from which test data are chosen.
            None means all data is used for training.
    Returns:
        The fitted XGBoost model for a player position."""

    xgb_model = XGBModel(df_pos, features, target)

    xgb_model.split_historical_data(test_split)

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

    base_model = MLModel(df, [target+Names.FORM_SUFFIX], target)

    base_model.split_historical_data(test_split)

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
    metrics['base'] = base_model.get_model_errors(y_pred)

    y_pred = test_model.model.predict(test_model.X_test)
    metrics['test'] = test_model.get_model_errors(y_pred)

    return metrics

def predict_position_points(df_features: pd.DataFrame, xgb_model: XGBModel, rows: pd.Series) -> pd.Series:
    """Predict player points using the XGBModel for a player position.
    Args:
        df_features: The pd.DataFrame of live features. Used for consistent indexing.
        xgb_model: The fitted XGBModel for a player position.
        rows: Boolean Series indicating rows of the player position.
    Returns:
        The pd.Series of the points predictions for the target rows."""

    X_test = df_features.loc[rows, xgb_model.features]
    return xgb_model.model.predict(X_test)
