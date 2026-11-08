
from typing import Any

import pandas as pd

from config.naming import Names
from config.settings import FPLSettings, Columns, ML_FEATURES_POS_FORM, PAST_SEASONS
from config.settings import POINTS_COEFFICIENTS, POINTS_COEFFICIENTS_POS, CURRENT_SEASON
from src.data import data_store
from src.models.ml_model import MLModel, XGBModel


def set_predictions(
    df_features: pd.DataFrame, fpl_settings: FPLSettings, store: bool = True
    ) -> pd.DataFrame:
    """Set players points predictions based on the analytical formula or a ML model.
    Args:
        df_features: The pd.DataFrame of live and past features.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A pd.DataFrame of live features and players points predictions."""

    if fpl_settings.prediction_model == Names.FORMULA:
        is_upcoming = (df_features[Columns.IS_FINISHED] == False)
        df_predict = df_features[is_upcoming]
        df_predict[Columns.POINTS_PREDICTED] = set_expected_points(df_predict)
    elif fpl_settings.prediction_model == Names.XGBOOST:
        df_predict = set_fpl_ml_predictions(df_features, fpl_settings.ml_target)

    col = Columns.POINTS_PREDICTED
    cols = [
        Columns.SEASON, Columns.PLAYER_CODE, Columns.PLAYER_COST, Columns.PLAYER_NAME,
        Columns.TEAM_NAME, Columns.POSITION_NAME, Columns.GAMEWEEK, col
        ]
    df_predict = df_predict[cols]
    if store:
        file_path = "fpl_data/predict/players_predictions.csv"
        data_store.save_df_to_csv(df_predict, file_path, overwrite=True)

    df_predict = df_predict.groupby(Columns.PLAYER_CODE, as_index=False).agg(
        player_name=(Columns.PLAYER_NAME, 'first'), position_name=(Columns.POSITION_NAME, 'first'),
        team_name=(Columns.TEAM_NAME, 'first'), player_cost=(Columns.PLAYER_COST, 'first'),
        PTS_predicted_sum=(Columns.POINTS_PREDICTED, 'sum'),
        )

    df_predict = df_predict.sort_values(by=col + '_sum', ascending=False)

    if store:
        file_path = "fpl_data/predict/players_predictions_sum.csv"
        data_store.save_df_to_csv(df_predict, file_path, overwrite=True)

    return df_predict

def set_expected_points(df: pd.DataFrame) -> pd.Series:
    """Set the expected players points based on expected values of the related quantities.
    Args:
        df: The pd.DataFrame of live features.
    Returns:
        A pd.Series of the expected players points."""

    off_team = df[Columns.OFFENSE_TEAM_FORM]
    def_team = df[Columns.DEFENSE_TEAM_FORM]
    off_opp = df[Columns.OFFENSE_OPPONENT_FORM]
    def_opp = df[Columns.DEFENSE_OPPONENT_FORM]

    xMIN = df[Columns.MINUTES_FORM]

    xG = df[Columns.EXPECTED_GOALS_FORM]*(off_team/def_opp)
    xA = df[Columns.EXPECTED_ASSISTS_FORM]*(off_team/def_opp)
    xGC = df[Columns.EXPECTED_GOALS_CONCEDED_FORM]*(off_opp/def_team)
    xCS = df[Columns.CLEAN_SHEETS_FORM]*(def_team/off_opp)
    xSA = df[Columns.SAVES_FORM]*off_opp
    xDC = df[Columns.DEFENSIVE_CONTRIBUTIONS_FORM]*off_opp
    xPS = df[Columns.PENALTIES_SAVED_FORM]
    xBP = df[Columns.BONUS_POINTS_FORM]
    xYC = df[Columns.YELLOW_CARDS_FORM]
    xRC = df[Columns.RED_CARDS_FORM]
    xOG = df[Columns.OWN_GOALS_FORM]
    xPM = df[Columns.PENALTIES_MISSED_FORM]

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

    xP = pd.Series(index=df.index, dtype=float)

    gkp_mask = df[Columns.POSITION_NAME] == Names.GOALKEEPER
    xP[gkp_mask] = cxG_gkp*xG - cxGC*xGC + cxSA*xSA + cxCS_gkp*xCS + xC

    def_mask = df[Columns.POSITION_NAME] == Names.DEFENDER
    xP[def_mask] = cxG_def*xG - cxGC*xGC + cxCS_def*xCS + xC
    xDC10 = xDC >= cxDC_bar_def
    xP[def_mask & xDC10] += cxDC

    mid_mask = df[Columns.POSITION_NAME] == Names.MIDFIELDER
    xP[mid_mask] = cxG_mid*xG + cxCS_mid*xCS + xC
    xDC12 = xDC >= cxDC_bar_mid
    xP[mid_mask & xDC12] += cxDC

    fwd_mask = df[Columns.POSITION_NAME] == Names.FORWARD
    xP[fwd_mask] = cxG_fwd*xG + xC
    xP[fwd_mask & xDC12] += cxDC

    return xP

def set_fpl_ml_predictions(df_features: pd.DataFrame, target: str) -> pd.DataFrame:
    """Set points predictions for each player based on a ML model for each player position.
    Args:
        df_features: The pd.DataFrame of features.
        target: The target column for ML predictions.
    Returns:
        A pd.DataFrame of the ML points predictions for each player."""

    is_upcoming = (df_features[Columns.IS_FINISHED] == False)
    df_predict = df_features[is_upcoming].copy()
    df_predict[Columns.POINTS_PREDICTED] = pd.Series(dtype=float)

    is_finished = (df_features[Columns.IS_FINISHED] == True)
    df_finished = df_features[is_finished]

    # test_split = (Columns.SEASON, PAST_SEASONS[-1])
    test_split = (Columns.SEASON, CURRENT_SEASON)
    for pos, ft in ML_FEATURES_POS_FORM.items():

        df_pos = df_finished[df_finished[Columns.POSITION_NAME] == pos]

        # compare with baseline (target form)
        base_model = set_base_model(df_pos, target, test_split)
        xgb_model = set_position_model(df_pos, ft, target, test_split)
        metrics = evaluate_position_model(xgb_model, base_model)

        compare_models(metrics, pos)

        # for key, value in metrics.items():
        #     print(f"{key}: {value}")

        # all data for training
        xgb_model = set_position_model(df_pos, ft, target, test_split=None)
        imp = xgb_model.get_features_importances()
        print(f"{pos} FEATURES IMPORTANCE\n{imp.sort_values(ascending=False)}")

        rows = (df_predict[Columns.POSITION_NAME] == pos)
        if rows.any():
            df_predict.loc[rows, Columns.POINTS_PREDICTED] = predict_position_points(df_predict, xgb_model, rows)

    return df_predict

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

    xgb_model.set_xgboost_model(n_estimators=100, max_depth=6, learning_rate=0.1)

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

def evaluate_position_model(xgb_model: XGBModel, base_model: XGBModel) -> dict[str, dict[str, float]]:
    """Evaluate the XGBoost model (fitted and baseline) metrics.
    Args:
        xgb_model: The fitted XGBModel instance.
        base_model: The baseline XGBModel instance.
    Returns:
        The dictionary of fitted and baseline model metrics."""

    metrics: dict[str, dict[str, float]] = {}

    col = base_model.features
    y_pred = base_model.X_test[col]
    metrics['base'] = base_model.get_model_errors(y_pred)

    y_pred = xgb_model.model.predict(xgb_model.X_test)
    metrics['xgb'] = xgb_model.get_model_errors(y_pred)

    return metrics

def compare_models(metrics: dict[str, dict[str, float]], pos: str) -> None:
    """Displays a side-by-side comparison table of Base Model vs. XGBoost."""

    df = pd.DataFrame(metrics)

    df.index = df.index.str.upper()

    df['change'] = df['base'] - df['xgb']
    df.loc['R2', 'change'] *= -1

    df['% change'] = ((df['change'])/df['base'])*100
    df.loc['R2', '% change'] = ((df.loc['R2', 'change'])/df.loc['R2', 'xgb'])*100

    print("\n" + "-"*50)
    print(f"{pos} MODEL COMPARISON")
    print("-"*50)
    print(df.to_string(float_format=lambda x: "  {:.4f}".format(x)))
    print("-"*50)

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
