
import pandas as pd
import numpy as np

from config import settings, naming
from src.data import data_store, data_features


def add_features(
    df_players: pd.DataFrame, players_past_data: dict[str, pd.DataFrame],
    fpl_settings: settings.FPLSettings, store: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add live and past features.
    Args:
        df_players: The pd.DataFrame of players live data.
        players_past_data: The dictionary of pd.DataFrames of players past data.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A tuple of pd.DataFrames of live and past features."""

    df_features = add_features_live(df_players=df_players, fpl_settings=fpl_settings)

    features_past = add_features_past(players_past_data=players_past_data, fpl_settings=fpl_settings)
    df_features_past = merge_past_seasons(features_past=features_past)
    df_features_past = prepare_features_past(df_features_past=df_features_past)

    if store:
        file_path = settings.LIVE_PLAYERS_STORE_FEATURES
        data_store.save_df_to_csv(df=df_features, file_path=file_path, overwrite=True)

        for s in features_past.keys():
            file_path = settings.PAST_STORE_FEATURES.format(season=s)
            data_store.save_df_to_csv(df=features_past[s], file_path=file_path, overwrite=True)

        file_path = settings.PAST_STORE_FEATURES_MERGED
        data_store.save_df_to_csv(df=df_features_past, file_path=file_path, overwrite=True)

    return df_features, df_features_past

def add_features_live(df_players: pd.DataFrame, fpl_settings: settings.FPLSettings) -> pd.DataFrame:
    """Add live features.
    Args:
        df_players: The pd.DataFrame of players live data.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
    Returns:
        A pd.DataFrame of live features."""

    fpl_features = data_features.DataFeatures(df=df_players)

    # Calculate per 90
    if fpl_settings.features_scale == 'form_per_90':

        min_form = settings.Columns.MINUTES + settings.Columns.FEATURES_SUFFIX
        features_form = [c for c in settings.FEATURES_FORM if c != min_form]
        dc_form = settings.Columns.DEFENSIVE_CONTRIBUTIONS + settings.Columns.FEATURES_SUFFIX
        if dc_form not in features_form:
            features_form.append(dc_form)
        fpl_features.calculate_per_x(columns=features_form, x_col=min_form, x=settings.MINUTES_PER_GAME)

        yc_bounds = data_features.Bounds(lower=0.0, upper=2.0)
        rc_bounds = data_features.Bounds(lower=0.0, upper=1.0)
        bp_bounds = data_features.Bounds(lower=0.0, upper=3.0)
        cs_bounds = data_features.Bounds(lower=0.0, upper=1.0)

        yc = settings.Columns.YELLOW_CARD + '_'+ fpl_settings.features_scale
        rc = settings.Columns.RED_CARD + '_'+ fpl_settings.features_scale
        bp = settings.Columns.BONUS_POINTS + '_'+ fpl_settings.features_scale
        cs = settings.Columns.CLEAN_SHEET + '_'+ fpl_settings.features_scale

        bounds_dict = {yc: yc_bounds, rc: rc_bounds, bp: bp_bounds, cs: cs_bounds}

        fpl_features.set_features_bounds(bounds_dict)

    df_features = fpl_features.df

    off_opp = settings.Columns.OFFENSE + '_' + settings.Columns.OPPONENT
    def_opp = settings.Columns.DEFENSE + '_' + settings.Columns.OPPONENT
    fdr_off = settings.Columns.FDR + '_' + settings.Columns.OFFENSE
    fdr_def = settings.Columns.FDR + '_' + settings.Columns.DEFENSE

    if (off_opp not in df_features.columns):
        metric = settings.Columns.EXPECTED_GOALS + settings.Columns.FEATURES_SUFFIX
        df_features[off_opp] = set_opponent_strength(df_features=df_features, metric=metric)

    if (def_opp not in df_features.columns):
        metric = settings.Columns.EXPECTED_GOALS_CONCEDED + settings.Columns.FEATURES_SUFFIX
        df_features[def_opp] = set_opponent_strength(df_features=df_features, metric=metric)

    df_features[fdr_off] = set_fixture_difficulty(df_features[def_opp])
    df_features[fdr_def] = set_fixture_difficulty(df_features[off_opp])

    df_features.drop(columns=[settings.Columns.IS_HOME, off_opp, def_opp], inplace=True)

    return df_features.copy()

def set_opponent_strength(df_features: pd.DataFrame, metric: str) -> pd.Series:
    """Set opponent strength based on opponent's specified metric.
    Args:
        df_features: The pd.DataFrame of live features.
        metric: The metric upon which strength is measured.
    Returns:
        A pd.Series[float] for the opponent's strength."""

    min_form = settings.Columns.MINUTES + settings.Columns.FEATURES_SUFFIX

    df_home = df_features[df_features[settings.Columns.IS_HOME]]
    sum_min_h = df_home.groupby(settings.Columns.TEAM_NAME)[min_form].sum().replace(0, np.nan)
    sum_strength_h = df_home.groupby(settings.Columns.TEAM_NAME)[metric].sum()

    df_away = df_features[~df_features[settings.Columns.IS_HOME]]
    sum_min_a = df_away.groupby(settings.Columns.TEAM_NAME)[min_form].sum().replace(0, np.nan)
    sum_strength_a = df_away.groupby(settings.Columns.TEAM_NAME)[metric].sum()

    strength_h = settings.MINUTES_PER_GAME*(sum_strength_h/sum_min_h)
    strength_a = settings.MINUTES_PER_GAME*(sum_strength_a/sum_min_a)

    # multiply xG by team players
    if (settings.Columns.EXPECTED_GOALS in metric and
        settings.Columns.EXPECTED_GOALS_CONCEDED not in metric):
        strength_h *= 11
        strength_a *= 11

    # inverse xGC
    if settings.Columns.EXPECTED_GOALS_CONCEDED in metric:
        strength_h = 1.0/strength_h.replace(0, np.nan)
        strength_a = 1.0/strength_a.replace(0, np.nan)

    col = settings.Columns.TEAM_NAME + '_' + settings.Columns.OPPONENT
    strength_h_opp = df_features[col].map(strength_h)
    strength_a_opp = df_features[col].map(strength_a)

    strength_opp = np.where(df_features[settings.Columns.IS_HOME], strength_a_opp, strength_h_opp)

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
    players_past_data: dict[str, pd.DataFrame], fpl_settings: settings.FPLSettings
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

        df_players.sort_values([settings.Columns.PLAYER_ID, settings.Columns.GW], inplace=True)

        features = [c.removesuffix(settings.Columns.FEATURES_SUFFIX) for c in settings.FEATURES_FORM]

        df_players[settings.FEATURES_FORM] = (
            df_players.groupby(settings.Columns.PLAYER_ID)[features]
            .transform(lambda s: s.rolling(window=fpl_settings.n_form).mean().shift(1))
        )

        cols_to_drop = [c for c in features if c != settings.Columns.POINTS]
        cols_to_drop.append(settings.Columns.PLAYER_ID)
        df_players.drop(columns=cols_to_drop, inplace=True)

        off_home_opp = settings.Columns.OFFENSE + '_' + naming.HOME + '_' + settings.Columns.OPPONENT
        off_away_opp = settings.Columns.OFFENSE + '_' + naming.AWAY + '_' + settings.Columns.OPPONENT
        def_home_opp = settings.Columns.DEFENSE + '_' + naming.HOME + '_' + settings.Columns.OPPONENT
        def_away_opp = settings.Columns.DEFENSE + '_' + naming.AWAY + '_' + settings.Columns.OPPONENT
        cols_opp = [settings.Columns.IS_HOME, off_home_opp, off_away_opp, def_home_opp, def_away_opp]
        are_present = set(cols_opp).issubset(df_players.columns)
        if are_present:
            fdr_off = settings.Columns.FDR + '_' + settings.Columns.OFFENSE
            df_players[fdr_off] = np.where(
                df_players[settings.Columns.IS_HOME], df_players[def_away_opp],
                df_players[def_home_opp])
            fdr_def = settings.Columns.FDR + '_' + settings.Columns.DEFENSE
            df_players[fdr_def] = np.where(
                df_players[settings.Columns.IS_HOME], df_players[off_away_opp],
                df_players[off_home_opp])
            df_players.drop(columns=cols_opp, inplace=True)
        else:
            df_players = add_features_live(df_players=df_players, fpl_settings=fpl_settings)

        past_features[s] = df_players.copy()

    return past_features

def merge_past_seasons(features_past: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge past features pd.DataFrames.
    Args:
        features_past: The dictionary of past features pd.DataFrames.
    Returns:
        A pd.DataFrame of the merged past features data."""

    df_list = []
    for s, df in features_past.items():
        df_new = df.copy()
        df_new[settings.Columns.SEASON] = s
        df_list.append(df_new)

    df_features_past = pd.concat(df_list)
    return df_features_past.copy()

def prepare_features_past(df_features_past: pd.DataFrame) -> pd.DataFrame:
    """Prepare the merged pd.DataFrame of past features for analysis.
    Args:
        df_features_past: The merged pd.DataFrame of past features.
    Returns:
        A pd.DataFrame of past features prepared for analysis."""

    fdr_off = settings.Columns.FDR + '_' + settings.Columns.OFFENSE
    fdr_def = settings.Columns.FDR + '_' + settings.Columns.DEFENSE

    columns_seasons = [
        settings.Columns.SEASON, settings.Columns.GW, settings.Columns.PLAYER_NAME,
        settings.Columns.POSITION_NAME, settings.Columns.TEAM_NAME,
        fdr_off, fdr_def,
        settings.Columns.POINTS
        ]
    for f in settings.FEATURES_FORM:
        columns_seasons.append(f)

    df = df_features_past[columns_seasons].copy()
    df = df.dropna()

    df.sort_values(
        by=[settings.Columns.SEASON, settings.Columns.GW,
            settings.Columns.TEAM_NAME, settings.Columns.POSITION_NAME],
        ascending=[True, True, True, True], inplace=True)

    return df.copy()
