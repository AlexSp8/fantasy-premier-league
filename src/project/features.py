
import pandas as pd
import numpy as np

from config import settings
from config.naming import Names
from config.settings import FPLSettings, Columns
from src.data import data_store, data_features


def add_features_live(
        df_players: pd.DataFrame, fpl_settings: FPLSettings, store: bool = True
        ) -> pd.DataFrame:
    """Add live features.
    Args:
        df_players: The pd.DataFrame of players live data.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
    Returns:
        A pd.DataFrame of live features."""

    fpl_features = data_features.DataFeatures(df=df_players)

    # Calculate per 90
    if fpl_settings.features_scale == Names.FORM_PER_90:

        min_form = Columns.MINUTES + Names.FORM_SUFFIX
        features_form = [c for c in settings.FEATURES_FORM if c != min_form]
        dc_form = Columns.DEFENSIVE_CONTRIBUTIONS + Names.FORM_SUFFIX
        if dc_form not in features_form:
            features_form.append(dc_form)
        fpl_features.calculate_per_x(features_form, min_form, settings.MINUTES_PER_GAME)

        yc_bounds = data_features.Bounds(lower=0.0, upper=2.0)
        rc_bounds = data_features.Bounds(lower=0.0, upper=1.0)
        bp_bounds = data_features.Bounds(lower=0.0, upper=3.0)
        cs_bounds = data_features.Bounds(lower=0.0, upper=1.0)

        yc = Columns.YELLOW_CARDS + Names.FORM_PER_90_SUFFIX
        rc = Columns.RED_CARDS + Names.FORM_PER_90_SUFFIX
        bp = Columns.BONUS_POINTS + Names.FORM_PER_90_SUFFIX
        cs = Columns.CLEAN_SHEET + Names.FORM_PER_90_SUFFIX

        bounds_dict = {yc: yc_bounds, rc: rc_bounds, bp: bp_bounds, cs: cs_bounds}

        fpl_features.set_features_bounds(bounds_dict)

    df_features = fpl_features.df

    off_opp = Names.OFFENSE + Names.OPPONENT_SUFFIX
    def_opp = Names.DEFENSE + Names.OPPONENT_SUFFIX

    if (off_opp not in df_features.columns):
        metric = Columns.EXPECTED_GOALS_FORM
        df_features[off_opp] = set_opponent_strength(df_features, metric)

    if (def_opp not in df_features.columns):
        metric = Columns.EXPECTED_GOALS_CONCEDED_FORM
        df_features[def_opp] = set_opponent_strength(df_features, metric)

    df_features[Columns.FDR_OFF] = set_fixture_difficulty(df_features[def_opp])
    df_features[Columns.FDR_DEF] = set_fixture_difficulty(df_features[off_opp])

    df_features.drop(columns=[Columns.IS_HOME, off_opp, def_opp], inplace=True)

    if store:
        data_store.save_df_to_csv(df_features, settings.STORE_FILE_PATH['features'][Names.LIVE], overwrite=True)

    return df_features

def set_opponent_strength(df_features: pd.DataFrame, metric: str) -> pd.Series:
    """Set opponent strength based on opponent's specified metric.
    Args:
        df_features: The pd.DataFrame of live features.
        metric: The metric upon which strength is measured.
    Returns:
        A pd.Series[float] for the opponent's strength."""

    df_home = df_features[df_features[Columns.IS_HOME]]
    sum_min_h = df_home.groupby(Columns.TEAM_NAME)[Columns.MINUTES_FORM].sum().replace(0, np.nan)
    sum_strength_h = df_home.groupby(Columns.TEAM_NAME)[metric].sum()

    df_away = df_features[~df_features[Columns.IS_HOME]]
    sum_min_a = df_away.groupby(Columns.TEAM_NAME)[Columns.MINUTES_FORM].sum().replace(0, np.nan)
    sum_strength_a = df_away.groupby(Columns.TEAM_NAME)[metric].sum()

    strength_h = settings.MINUTES_PER_GAME*(sum_strength_h/sum_min_h)
    strength_a = settings.MINUTES_PER_GAME*(sum_strength_a/sum_min_a)

    # multiply xG by team players
    if (Columns.EXPECTED_GOALS in metric and
        Columns.EXPECTED_GOALS_CONCEDED not in metric):
        strength_h *= 11
        strength_a *= 11

    # inverse xGC
    if Columns.EXPECTED_GOALS_CONCEDED in metric:
        strength_h = 1.0/strength_h.replace(0, np.nan)
        strength_a = 1.0/strength_a.replace(0, np.nan)

    col = Columns.TEAM_NAME + Names.OPPONENT_SUFFIX
    strength_h_opp = df_features[col].map(strength_h)
    strength_a_opp = df_features[col].map(strength_a)

    strength_opp = np.where(df_features[Columns.IS_HOME], strength_a_opp, strength_h_opp)

    return pd.Series(strength_opp, index=df_features.index)

def set_fixture_difficulty(strength_opp: pd.Series) -> pd.Series:
    """Set fixture difficulty based on the opponent's strength.
    Args:
        strength_opp: The pd.Series of opponent strength feature.
    Returns:
        A pd.Series of fixture difficulty rating."""

    strength_mean = strength_opp.mean()

    fdr = strength_opp.replace(0, np.nan)/strength_mean

    fdr = fdr.fillna(1.0)
    # Max 50% variation
    fdr = fdr.clip(lower=0.5, upper=1.5)

    return fdr

def add_features_past(
        players_past_data: dict[str, pd.DataFrame], fpl_settings: FPLSettings, store: bool = True
        ) -> dict[str, pd.DataFrame]:
    """Add past features.
    Args:
        players_past_data: The dictionary of pd.DataFrame of players past data.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
    Returns:
        A dictionary of pd.DataFrames of past features."""

    past_features: dict[str, pd.DataFrame] = {}
    for s in players_past_data.keys():

        df_players = players_past_data[s].copy()

        df_players.sort_values([Columns.PLAYER_ID, Columns.GW], inplace=True)

        features = [c for c in settings.FEATURES]

        df_players[settings.FEATURES_FORM] = (
            df_players.groupby(Columns.PLAYER_ID)[features]
            .transform(lambda s: s.rolling(window=fpl_settings.n_form).mean().shift(1))
        )

        cols_to_drop = [c for c in features if c != Columns.POINTS]
        cols_to_drop.append(Columns.PLAYER_ID)
        df_players.drop(columns=cols_to_drop, inplace=True)

        off_home_opp = Names.OFFENSE + Names.HOME_SUFFIX + Names.OPPONENT_SUFFIX
        off_away_opp = Names.OFFENSE + Names.AWAY_SUFFIX + Names.OPPONENT_SUFFIX
        def_home_opp = Names.DEFENSE + Names.HOME_SUFFIX + Names.OPPONENT_SUFFIX
        def_away_opp = Names.DEFENSE + Names.AWAY_SUFFIX + Names.OPPONENT_SUFFIX
        cols_opp = [Columns.IS_HOME, off_home_opp, off_away_opp, def_home_opp, def_away_opp]
        are_present = set(cols_opp).issubset(df_players.columns)
        if are_present:
            fdr_off = Columns.FDR_OFF
            df_players[fdr_off] = np.where(
                df_players[Columns.IS_HOME], df_players[def_away_opp], df_players[def_home_opp]
                )
            fdr_def = Columns.FDR_DEF
            df_players[fdr_def] = np.where(
                df_players[Columns.IS_HOME], df_players[off_away_opp], df_players[off_home_opp]
                )
            df_players.drop(columns=cols_opp, inplace=True)
        else:
            df_players = add_features_live(df_players, fpl_settings, store=False)

        past_features[s] = df_players.copy()

    df_features_past = merge_past_seasons(past_features)
    df_features_past = prepare_features_past(df_features_past)

    if store:
        data_store.save_df_to_csv(df_features_past, settings.STORE_FILE_PATH['features']['past_merged'], overwrite=True)
        for s in past_features.keys():
            file_path = settings.STORE_FILE_PATH['features']['past'].format(season=s)
            data_store.save_df_to_csv(past_features[s], file_path, overwrite=True)

    return df_features_past

def merge_past_seasons(features_past: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge past features pd.DataFrames.
    Args:
        features_past: The dictionary of past features pd.DataFrames.
    Returns:
        A pd.DataFrame of the merged past features data."""

    df_list = []
    for s, df in features_past.items():
        df_new = df.copy()
        df_new[Columns.SEASON] = s
        df_list.append(df_new)

    df_features_past = pd.concat(df_list)
    return df_features_past.copy()

def prepare_features_past(df_features_past: pd.DataFrame) -> pd.DataFrame:
    """Prepare the merged pd.DataFrame of past features for analysis.
    Args:
        df_features_past: The merged pd.DataFrame of past features.
    Returns:
        A pd.DataFrame of past features prepared for analysis."""

    columns_seasons = [
        Columns.SEASON, Columns.GW, Columns.PLAYER_NAME,
        Columns.POSITION_NAME, Columns.TEAM_NAME, Columns.FDR_OFF, Columns.FDR_DEF,
        Columns.POINTS
        ]
    for f in settings.FEATURES_FORM:
        columns_seasons.append(f)

    df = df_features_past[columns_seasons].copy()
    df = df.dropna()

    df.sort_values(
        by=[Columns.SEASON, Columns.GW, Columns.TEAM_NAME, Columns.POSITION_NAME],
        ascending=[True, True, True, True], inplace=True
        )

    return df.copy()
