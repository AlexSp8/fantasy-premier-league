
import pandas as pd
import numpy as np

from config import settings
from config.naming import Names
from config.settings import FPLSettings, Columns
from src.data import data_store, data_features


def add_features(
        players_data: dict[str, pd.DataFrame], fpl_settings: FPLSettings, store: bool = True
        ) -> pd.DataFrame:
    """Add players features.
    Args:
        players_data: The dictionary of pd.DataFrame of players data.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
    Returns:
        A pd.DataFrame of seasons-merged features."""

    features = settings.FEATURES.copy()
    features_form = settings.FEATURES_FORM.copy()
    dc_missing = Columns.DEFENSIVE_CONTRIBUTIONS not in features
    if fpl_settings.prediction_model == Names.FORMULA and dc_missing:
        features.append(Columns.DEFENSIVE_CONTRIBUTIONS)
        features_form.append(Columns.DEFENSIVE_CONTRIBUTIONS_FORM)

    players_features: dict[str, pd.DataFrame] = {}
    for s in players_data.keys():

        df = players_data[s].copy()

        df.sort_values([Columns.GAMEWEEK, Columns.TEAM_NAME, Columns.POSITION_NAME], inplace=True)

        group_col = Columns.PLAYER_ID
        if Columns.PLAYER_CODE in df.columns:
            group_col = Columns.PLAYER_CODE

        df[features_form] = (
            df.groupby(group_col)[features]
            .transform(lambda s: s.rolling(window=fpl_settings.n_form).mean().shift(1))
        )
        # forward fill upcoming fixtures
        df[features_form] = (df.groupby(group_col)[features_form].transform(lambda s: s.ffill()))

        df = add_fixtures_features(df)

        if fpl_settings.features_scale == Names.FORM_PER_90 and fpl_settings.prediction_model != Names.FORMULA:
            df = add_per_90_features(df, features_form)
            features_form = settings.FEATURES_FORM_PER_90.copy()
            if fpl_settings.prediction_model == Names.FORMULA and dc_missing:
                features_form.append(Columns.DEFENSIVE_CONTRIBUTIONS_FORM + Names.FORM_PER_90_SUFFIX)

        cols_to_drop = [c for c in features if c != Columns.POINTS]
        df.drop(columns=cols_to_drop, inplace=True)

        # off_home_opp = Names.OFFENSE + Names.HOME_SUFFIX + Names.OPPONENT_SUFFIX
        # off_away_opp = Names.OFFENSE + Names.AWAY_SUFFIX + Names.OPPONENT_SUFFIX
        # def_home_opp = Names.DEFENSE + Names.HOME_SUFFIX + Names.OPPONENT_SUFFIX
        # def_away_opp = Names.DEFENSE + Names.AWAY_SUFFIX + Names.OPPONENT_SUFFIX
        # cols_opp = [Columns.IS_HOME, off_home_opp, off_away_opp, def_home_opp, def_away_opp]
        # are_present = set(cols_opp).issubset(df.columns)
        # if are_present:
        #     df[Columns.FDR_OFF] = np.where(df[Columns.IS_HOME], df[def_away_opp], df[def_home_opp])
        #     df[Columns.FDR_DEF] = np.where(df[Columns.IS_HOME], df[off_away_opp], df[off_home_opp])
        #     df.drop(columns=cols_opp, inplace=True)

        players_features[s] = df.copy()

    df_merged = merge_seasons(players_features, features_form)

    if store:
        data_store.save_df_to_csv(df_merged, settings.STORE_FILE_PATH[Names.FEATURES]['merged'], overwrite=True)
        for s in players_features.keys():
            file_path = settings.STORE_FILE_PATH[Names.FEATURES][Names.PAST].format(season=s)
            data_store.save_df_to_csv(players_features[s], file_path, overwrite=True)

    return df_merged

def add_per_90_features(df_players: pd.DataFrame, features_form: list[str]) -> pd.DataFrame:
    """Add per-90 features.
    Args:
        df_players: The pd.DataFrame of players data.
        features_form: The list of features form names.
    Returns:
        A pd.DataFrame of per-90 features."""

    fpl_features = data_features.DataFeatures(df=df_players)

    min_form = Columns.MINUTES_FORM
    # features_form = [c for c in features_form if c != min_form]
    fpl_features.calculate_per_x(features_form, min_form, settings.MINUTES_PER_GAME)

    yc_bounds = data_features.Bounds(lower=0.0, upper=2.0)
    rc_bounds = data_features.Bounds(lower=0.0, upper=1.0)
    bp_bounds = data_features.Bounds(lower=0.0, upper=3.0)
    cs_bounds = data_features.Bounds(lower=0.0, upper=1.0)

    yc = Columns.YELLOW_CARDS + Names.FORM_PER_90_SUFFIX
    rc = Columns.RED_CARDS + Names.FORM_PER_90_SUFFIX
    bp = Columns.BONUS_POINTS + Names.FORM_PER_90_SUFFIX
    cs = Columns.CLEAN_SHEETS + Names.FORM_PER_90_SUFFIX

    bounds_dict = {yc: yc_bounds, rc: rc_bounds, bp: bp_bounds, cs: cs_bounds}

    fpl_features.set_features_bounds(bounds_dict)

    fpl_features.df.drop(columns=features_form, inplace=True)

    return fpl_features.df

def add_fixtures_features(df_players: pd.DataFrame) -> pd.DataFrame:
    """Add fixtures features.
    Args:
        df_players: The pd.DataFrame of players data.
    Returns:
        A pd.DataFrame with added fixtures features."""

    df_features = df_players.copy()

    off_team = Columns.OFFENSE_TEAM_FORM
    if (off_team not in df_features.columns):
        metric = Columns.EXPECTED_GOALS_FORM
        off_strength = set_team_strength(df_features, metric, opponent=False)
        df_features[Columns.OFFENSE_TEAM_FORM] = normalize_feature(off_strength)

    def_team = Columns.DEFENSE_TEAM_FORM
    if (def_team not in df_features.columns):
        metric = Columns.EXPECTED_GOALS_CONCEDED_FORM
        def_strength = set_team_strength(df_features, metric, opponent=False)
        df_features[Columns.DEFENSE_TEAM_FORM] = normalize_feature(def_strength)

    off_opp = Columns.OFFENSE_OPPONENT_FORM
    if (off_opp not in df_features.columns):
        metric = Columns.EXPECTED_GOALS_FORM
        off_strength_opp = set_team_strength(df_features, metric, opponent=True)
        df_features[Columns.OFFENSE_OPPONENT_FORM] = normalize_feature(off_strength_opp)

    def_opp = Columns.DEFENSE_OPPONENT_FORM
    if (def_opp not in df_features.columns):
        metric = Columns.EXPECTED_GOALS_CONCEDED_FORM
        def_strength_opp = set_team_strength(df_features, metric, opponent=True)
        df_features[Columns.DEFENSE_OPPONENT_FORM] = normalize_feature(def_strength_opp)

    # df_features.drop(columns=[Columns.IS_HOME, off_team, def_team, off_opp, def_opp], inplace=True)

    return df_features

def set_team_strength(df_features: pd.DataFrame, metric: str, opponent: bool) -> pd.Series:
    """Set team strength based on specified metric.
    Args:
        df_features: The pd.DataFrame of live features.
        metric: The metric upon which strength is measured.
        opponent: Boolean to set opponent strength or not.
    Returns:
        A pd.Series[float] for the team strength."""

    group_keys = [Columns.TEAM_NAME, Columns.GAMEWEEK]

    sum_min = df_features.groupby(group_keys)[Columns.MINUTES_FORM].sum().replace(0, np.nan)
    sum_strength = df_features.groupby(group_keys)[metric].sum()

    # inverse xGC
    if Columns.EXPECTED_GOALS_CONCEDED in metric:
        sum_strength = 1.0/sum_strength.replace(0, np.nan)

    strength = 11*settings.MINUTES_PER_GAME*(sum_strength/sum_min)

    team_col = Columns.TEAM_NAME
    if opponent:
        team_col = Columns.TEAM_NAME + Names.OPPONENT_SUFFIX

    temp_df = df_features[[team_col, Columns.GAMEWEEK]].copy()

    strength = temp_df.set_index([team_col, Columns.GAMEWEEK]).index.map(strength)

    return pd.Series(strength, index=df_features.index)

def normalize_feature(feature: pd.Series) -> pd.Series:
    """Normalize a feature using its mean value.
    Args:
        feature: The pd.Series of feature to be normalized.
    Returns:
        A pd.Series of normalized feature."""

    feature_mean = feature.mean()

    feature_norm = feature.replace(0, np.nan)/feature_mean

    feature_norm = feature_norm.fillna(1.0)
    # # Max 50% variation
    # feature_norm = feature_norm.clip(lower=0.5, upper=1.5)

    return feature_norm

def merge_seasons(players_features: dict[str, pd.DataFrame], features_form: list[str]) -> pd.DataFrame:
    """Merge seasons features pd.DataFrames.
    Args:
        players_features: The dictionary of features pd.DataFrames.
        features_form: The list of features form names.
    Returns:
        A pd.DataFrame of the merged past features data."""

    df_list = []
    for s, df in players_features.items():
        df_new = df.copy()
        df_new[Columns.SEASON] = s
        df_list.append(df_new)

    df = pd.concat(df_list)

    columns_seasons = [
        Columns.SEASON, Columns.PLAYER_CODE, Columns.PLAYER_COST, Columns.PLAYER_NAME,
        Columns.POSITION_NAME, Columns.TEAM_NAME, Columns.GAMEWEEK, Columns.IS_FINISHED,
        Columns.OFFENSE_TEAM_FORM, Columns.DEFENSE_TEAM_FORM,
        Columns.OFFENSE_OPPONENT_FORM, Columns.DEFENSE_OPPONENT_FORM,
        Columns.POINTS
        ]
    for f in features_form:
        columns_seasons.append(f)

    df = df[columns_seasons]
    # df.drop(columns=[Columns.PLAYER_ID], inplace=True)
    df = df.dropna(subset=features_form)

    df.sort_values(
        by=[Columns.SEASON, Columns.GAMEWEEK, Columns.TEAM_NAME, Columns.POSITION_NAME],
        ascending=[False, True, True, True], inplace=True
        )

    return df
