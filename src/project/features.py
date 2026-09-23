
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from config.settings import FEATURES_FORM, MINUTES_PER_GAME
from src.data import data_store, data_features


def add_features(
    df_players: pd.DataFrame, players_past_data: dict[str, pd.DataFrame],
    n_form: int = 1, store: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add live and past features.
    Args:
        df_players: The pd.DataFrame of players live data.
        players_past_data: The dictionary of pd.DataFrames of players past data.
        n_form (optional): The size of the past rolling window.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A tuple of pd.DataFrames of live and past features."""

    df_features = add_features_live(df_players=df_players)

    features_past = add_features_past(players_past_data=players_past_data, n_form=n_form)
    df_features_past = merge_past_seasons(features_past=features_past)
    df_features_past = prepare_features_past(df_features_past=df_features_past)

    if store:
        file_path = f"fpl_data/clean/players_data_features.csv"
        data_store.save_df_to_csv(df=df_features, file_path=file_path, overwrite=True)

        for s in features_past.keys():
            file_path = f"fpl_data/past/clean/season_{s}_features.csv"
            data_store.save_df_to_csv(df=features_past[s], file_path=file_path, overwrite=True)

        file_path = f"fpl_data/past/clean/past_seasons_features_merged.csv"
        data_store.save_df_to_csv(df=df_features_past, file_path=file_path, overwrite=True)
        # print(df.info())

    return df_features, df_features_past

def add_features_live(df_players: pd.DataFrame) -> pd.DataFrame:
    """Add live features.
    Args:
        df_players: The pd.DataFrame of players live data.
    Returns:
        A pd.DataFrame of live features."""

    fpl_features = data_features.DataFeatures(df=df_players)

    # # Calculate per 90
    # features_form = FEATURES_FORM.copy()
    # if 'DC_form' not in features_form:
    #     features_form.append('DC_form')
    # features_form.remove('MIN_form')
    # fpl_features.calculate_per_x(columns=features_form, x_col='MIN_form', x=MINUTES_PER_GAME)

    # yc_bounds = data_features.Bounds(lower=0.0, upper=2.0)
    # rc_bounds = data_features.Bounds(lower=0.0, upper=1.0)
    # bp_bounds = data_features.Bounds(lower=0.0, upper=3.0)
    # cs_bounds = data_features.Bounds(lower=0.0, upper=1.0)
    # bounds_dict = {
    #     'YC_form_per_90': yc_bounds, 'RC_form_per_90': rc_bounds,
    #     'BP_form_per_90': bp_bounds, 'CS_form_per_90': cs_bounds
    #     }
    # fpl_features.set_features_bounds(bounds_dict)

    df_features = fpl_features.df

    if ('off_opp' not in df_features.columns):
        df_features['off_opp'] = set_opponent_strength(df_features=df_features, metric='xG_form')

    if ('def_opp' not in df_features.columns):
        df_features['def_opp'] = set_opponent_strength(df_features=df_features, metric='xGC_form')

    df_features['fdr_off'] = set_fixture_difficulty(df_features['def_opp'])
    df_features['fdr_def'] = set_fixture_difficulty(df_features['off_opp'])

    # print(df_features[columns].isna().sum())

    df_features.drop(columns=['is_home', 'off_opp', 'def_opp'], inplace=True)

    return df_features.copy()

def set_opponent_strength(df_features: pd.DataFrame, metric: str) -> pd.Series[float]:
    """Set opponent strength based on opponent's specified metric.
    Args:
        df_features: The pd.DataFrame of live features.
        metric: The metric upon which strength is measured.
    Returns:
        A pd.Series[float] for the opponent's strength."""

    df_home = df_features[df_features['is_home']]
    sum_min_h = df_home.groupby('team_name')['MIN_form'].sum().replace(0, np.nan)
    sum_strength_h = df_home.groupby('team_name')[metric].sum()

    df_away = df_features[~df_features['is_home']]
    sum_min_a = df_away.groupby('team_name')['MIN_form'].sum().replace(0, np.nan)
    sum_strength_a = df_away.groupby('team_name')[metric].sum()

    strength_h = MINUTES_PER_GAME*(sum_strength_h/sum_min_h)
    strength_a = MINUTES_PER_GAME*(sum_strength_a/sum_min_a)

    # multiply xG by team players
    if 'xG' in metric and 'xGC' not in metric:
        strength_h *= 11
        strength_a *= 11

    # inverse xGC
    if 'xGC' in metric:
        strength_h = 1.0/strength_h.replace(0, np.nan)
        strength_a = 1.0/strength_a.replace(0, np.nan)

    strength_h_opp = df_features['team_name_opp'].map(strength_h)
    strength_a_opp = df_features['team_name_opp'].map(strength_a)

    strength_opp = np.where(df_features['is_home'], strength_a_opp, strength_h_opp)

    return pd.Series(strength_opp, index=df_features.index)

def set_fixture_difficulty(strength_opp: pd.Series) -> NDArray:
    """Set fixture difficulty based on the opponent's strength.
    Args:
        strength_opp: The pd.Series of opponent strength feature.
    Returns:
        A NDArray of fixture difficulty rating."""

    strength_mean = strength_opp.mean()

    fdr = strength_opp.replace(0, np.nan)/strength_mean

    fdr = fdr.fillna(1.0)
    # Max 50% variation
    fdr = fdr.clip(lower=0.5, upper=1.5)

    return fdr

def add_features_past(players_past_data: dict[str, pd.DataFrame], n_form: int = 1) -> dict[str, pd.DataFrame]:
    """Add past features.
    Args:
        players_past_data: The dictionary of pd.DataFrame of players past data.
        n_form: The size of the past rolling window.
    Returns:
        A dictionary of pd.DataFrames of past features."""

    past_features: dict[str, pd.DataFrame] = {}
    for s in players_past_data.keys():

        df_players = players_past_data[s].copy()

        df_players.sort_values(['player_id', 'GW'], inplace=True)

        features = [c.removesuffix('_form') for c in FEATURES_FORM]

        df_players[FEATURES_FORM] = (
            df_players.groupby('player_id')[features]
            .transform(lambda s: s.rolling(window=n_form).mean().shift(1))
        )

        cols_to_drop = [c for c in features if c != 'PTS']
        cols_to_drop.append('player_id')
        df_players.drop(columns=cols_to_drop, inplace=True)

        cols_opp = ['is_home', 'off_home_opp', 'off_away_opp', 'def_home_opp','def_away_opp']
        are_present = set(cols_opp).issubset(df_players.columns)
        if are_present:
            df_players['fdr_off'] = np.where(
                df_players['is_home'], df_players['def_away_opp'],
                df_players['def_home_opp'])
            df_players['fdr_def'] = np.where(
                df_players['is_home'], df_players['off_away_opp'],
                df_players['off_home_opp'])
            df_players.drop(columns=cols_opp, inplace=True)
        else:
            df_players = add_features_live(df_players=df_players)

        # df_players.dropna(inplace=True)

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
        df_new['season'] = s
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
        'season', 'GW', 'player_name', 'position_name', 'team_name',
        'fdr_off', 'fdr_def',
        'PTS'
        ]
    for f in FEATURES_FORM:
        columns_seasons.append(f)

    df = df_features_past[columns_seasons].copy()
    # print(df.isna().sum())
    df = df.dropna()

    df = df.sort_values(by=['season', 'GW', 'team_name', 'position_name'], ascending=[True, True, True, True])

    return df.copy()
