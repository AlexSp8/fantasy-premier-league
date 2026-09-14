
from typing import Any
import time

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from config import settings
from config.settings import FPLSettings
from src.data import data_fetch, data_store, data_cleaner, data_features
from src.models import fpl_model


def fetch_data(
    fpl_settings: FPLSettings, store: bool = True
    ) -> tuple[dict[str, Any], dict[str, dict[str, pd.DataFrame]]]:
    """Fetch live (current season) data (players, teams, fixtures, GW history)
    and past (seasons) data (players, teams) from API or existing directory.
    Args:
        fpl_settings: The FPLSettings object specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not. Defaults to True.
    Returns:
        A tuple of a dictionary of players, teams, fixtures, GW history live data
        and of a dictionary of players and teams past data."""

    live_data = fetch_live_data(fpl_settings)

    # if settings.PREDICTION_MODEL != "formula":
    past_data = fetch_past_data(fpl_settings)

    if store:
        store_data_raw(live_data=live_data, past_data=past_data)

    return live_data, past_data

def fetch_live_data(fpl_settings: FPLSettings) -> dict[str, Any]:
    """Fetch live data (players, teams, fixtures, GW history) from API or existing directory.
    Args:
        fpl_settings: The FPLSettings object specifying constraints and settings.
    Returns:
        A dictionary of players, teams, fixtures, GW history live data."""

    # Players, Teams
    live_data: dict[str, Any] = data_fetch.fetch_json_data(
        url=fpl_settings.live_url, file_path="fpl_data/raw/fpl_data_raw.json"
        )

    key = "fixtures"
    if key not in live_data.keys():
        live_data[key] = data_fetch.fetch_json_data(url=fpl_settings.fixtures_url)

    key = "gw_history"
    if key not in live_data.keys():
        player_ids = [row['id'] for row in live_data["elements"]]
        # player_ids = [i for i in range(1,6)]
        live_data[key] = fetch_live_gw_data(player_ids=player_ids, fpl_settings=fpl_settings)

    return live_data

def fetch_live_gw_data(player_ids: list[int], fpl_settings: FPLSettings) -> list[list[dict[str, Any]]]:
    """Fetch players GW history live data.
    Args:
        players_ids: The list of live player ids.
        fpl_settings: The FPLSettings object specifying constraints and settings.
    Returns:
        A list of lists of dictionaries of players GW history live data."""

    gw_history: list[list[dict[str, Any]]] = []

    player_ids_sorted = player_ids
    player_ids_sorted.sort()

    failed_ids = []
    for pid in player_ids_sorted:
        try:
            url = fpl_settings.gw_url + str(pid)
            player_history: dict[str, Any] = data_fetch.fetch_json_data(url=url)
            player_gw_history: list[dict[str, Any]] = player_history['history']
            time.sleep(0.5)
            gw_history.append(player_gw_history)
        except Exception as e:
            print(f"Failed to fetch data for player {pid}. Exception: {e}")
            failed_ids.append(pid)

    return gw_history

def fetch_past_data(fpl_settings: FPLSettings) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch past data (players, teams) from API or existing directory.
    Args:
        fpl_settings: The FPLSettings object specifying constraints and settings.
    Returns:
        A dictionary of pd.DataFrames of players, teams past data."""

    file_paths = []
    file_paths_teams = []
    for s in settings.PAST_SEASONS:
        file_paths.append(f"fpl_data/past/raw/season_{s}_raw.csv")
        file_paths_teams.append(f"fpl_data/past/raw/season_{s}_teams_raw.csv")

    prefix = fpl_settings.past_url
    suffix = fpl_settings.past_players_suffix
    past_players_data = data_fetch.fetch_multiple_csv_data(
        prefix=prefix, suffix=suffix, identifier=settings.PAST_SEASONS, file_paths=file_paths
        )

    suffix = fpl_settings.past_teams_suffix
    past_teams_data = data_fetch.fetch_multiple_csv_data(
        prefix=prefix, suffix=suffix, identifier=settings.PAST_SEASONS, file_paths=file_paths_teams
        )

    past_data = {'players': past_players_data, 'teams': past_teams_data}

    return past_data

def store_data_raw(live_data: dict[str, Any], past_data: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Store live and past raw data.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data.
        past_data: The dictionary of players, teams past data."""

    data_store.store_data_to_json(data=live_data, file_path="fpl_data/raw/fpl_data_raw.json")

    data_store.save_list_dict_to_csv(
        data_list_dict=live_data["elements"], file_path="fpl_data/raw/players_data_raw.csv"
        )

    data_store.save_list_dict_to_csv(
        data_list_dict=live_data["teams"], file_path="fpl_data/raw/teams_data_raw.csv"
        )

    data_store.save_list_dict_to_csv(
        data_list_dict=live_data["fixtures"], file_path= "fpl_data/raw/fixtures_data_raw.csv"
        )

    df_gw_list = [pd.DataFrame(data=player) for player in live_data["gw_history"]]
    df_players_gw = pd.concat(df_gw_list, ignore_index=True)
    data_store.save_df_to_csv(df=df_players_gw, file_path= "fpl_data/raw/players_gw_data_raw.csv")

    for s, df in past_data['players'].items():
        file_path = f"fpl_data/past/raw/season_{s}_raw.csv"
        data_store.save_df_to_csv(df=df, file_path=file_path)
    for s, df in past_data['teams'].items():
        file_path = f"fpl_data/past/raw/season_{s}_teams_raw.csv"
        data_store.save_df_to_csv(df=df, file_path=file_path)

def prepare_players_data(
    live_data: dict[str, Any], past_data: dict[str, dict[str, pd.DataFrame]],
    fpl_settings: FPLSettings, store: bool = True
    ) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Prepare players live and past data for analysis.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data.
        past_data: The dictionary of players, teams past data.
        fpl_settings: The FPLSettings object specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not. Defaults to True.
    Returns:
        A tuple of a pd.DataFrame of players live data and
        of a dictionary of pd.DataFrames of players past data."""

    df_players = prepare_players_live_data(live_data=live_data, fpl_settings=fpl_settings)

    players_past_data = prepare_players_past_data(past_data=past_data)

    if store:
        file_path = f"fpl_data/clean/players_data_clean.csv"
        data_store.save_df_to_csv(df=df_players, file_path=file_path)
        for s in players_past_data.keys():
            file_path = f"fpl_data/past/clean/season_{s}_clean.csv"
            data_store.save_df_to_csv(df=players_past_data[s], file_path=file_path)

    return df_players, players_past_data

def prepare_players_live_data(live_data: dict[str, Any], fpl_settings: FPLSettings) -> pd.DataFrame:
    """Prepare players live data for analysis.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data.
        fpl_settings: The FPLSettings object specifying constraints and settings.
    Returns:
        A pd.DataFrame of players live data prepared for analysis."""

    df_players = pd.DataFrame(data=live_data["elements"]).copy()

    df_players = df_players[settings.PLAYERS_COLUMNS_KEEP]

    available_players = ((df_players['can_select']) & (df_players['status'] != 'u') &
                         (df_players['status'] != 'i') & (df_players['status'] != 's'))

    df_players = df_players[available_players].rename(columns=settings.PLAYERS_COLUMNS_RENAME)

    rows = (df_players['status'] == 'a')
    column = 'availability'
    df_players.loc[rows, column] = 100

    df_players = df_players[df_players['availability'] == 100]

    df_players['position_name'] = df_players['element_type'].map(settings.POSITION_MAP)

    df_teams = pd.DataFrame(data=live_data["teams"]).copy()
    df_teams = df_teams[settings.TEAMS_COLUMNS_KEEP]
    df_teams = df_teams.rename(columns=settings.TEAMS_COLUMNS_RENAME)

    df_players = df_players.merge(df_teams, on='team_id', how="inner")

    df_fixtures = pd.DataFrame(data=live_data["fixtures"]).copy()
    df_fixtures = df_fixtures[settings.FIXTURES_COLUMNS_KEEP]
    df_fixtures = df_fixtures.rename(columns=settings.FIXTURES_COLUMNS_RENAME)
    df_fixtures = df_fixtures[~df_fixtures['finished']]
    df_fixtures = df_fixtures.drop(columns=['finished'])

    df_home = df_fixtures.rename(columns={'team_a': 'opponent_id', 'team_h': 'team_id'})
    df_home['is_home'] = True

    df_away = df_fixtures.rename(columns={'team_h': 'opponent_id', 'team_a': 'team_id'})
    df_away['is_home'] = False

    df_home_away = pd.concat([df_home, df_away], ignore_index=True)
    df_home_away = (df_home_away.sort_values(by='GW', ascending=True)
                    .groupby('team_id').head(fpl_settings.n_fixtures))

    df_players = df_players.merge(df_home_away, on='team_id', how="inner")

    df_players = df_players.merge(df_teams, left_on="opponent_id", right_on='team_id', how="inner",
                              suffixes=("", "_opponent"))

    df_players = df_players.sort_values(by=['GW', 'team_name', 'element_type'], ascending=[True, True, True])

    df_list = [pd.DataFrame(data=player) for player in live_data["gw_history"]].copy()
    df_players_gw_raw = pd.concat(df_list, ignore_index=True)
    df_players_gw = df_players_gw_raw[settings.PLAYERS_GW_COLUMNS_KEEP]
    df_players_gw = df_players_gw.rename(columns=settings.PLAYERS_GW_COLUMNS_RENAME)
    df_players_gw = (df_players_gw.sort_values(by=['GW', 'player_id'], ascending=True)
                     .groupby('player_id').tail(fpl_settings.n_form))
    cleaner = data_cleaner.DataCleaner(df=df_players_gw)
    cleaner.clean_data()
    df_players_gw = cleaner.df.copy()

    features_form = settings.FEATURES_FORM.copy()
    if 'DC' not in features_form:
        features_form.append('DC')
    # for c in features_form:
    #     df_players_gw[c+'_form'] = (
    #         df_players_gw.groupby('player_id')[c].transform('mean')
    #         )
    # df_players_gw = df_players_gw.drop(columns=features_form)

    df_players_gw = (
        df_players_gw.groupby('player_id')[features_form]
        .mean()
        .add_suffix('_form')
        .reset_index()
    )

    # file_path = f"fpl_data/clean/players_gw_data_clean.csv"
    # data_store.save_df_to_csv(df=df_players_gw, file_path=file_path)

    df_players = df_players.merge(df_players_gw, on='player_id', how="left")

    df_players = df_players.drop(columns=[
        'availability', 'can_select', 'status', 'team_id', 'element_type', 'opponent_id',
        'team_id_opponent',
        # 'OFF_home', 'OFF_away', 'DEF_home', 'DEF_away'
        ])

    cleaner = data_cleaner.DataCleaner(df=df_players)
    cleaner.clean_data()

    df_negative = cleaner.flag_negative_values(
        excluded_keywords=settings.EXCLUDED_FROM_NEGATIVE_CHECK
        )
    if not df_negative.empty:
        raise ValueError("DataFrame contains incorrect negative values!")

    df_players = cleaner.df.copy()

    return df_players

def prepare_players_past_data(past_data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Prepare players past data for analysis.
    Args:
        past_data: The dictionary of players, teams past data.
    Returns:
        A dictionary of pd.DataFrames of players past data prepared for analysis."""

    players_past_data: dict[str, pd.DataFrame] = {}

    for s in past_data['players'].keys():

        df_players = past_data['players'][s].copy()

        df_players = df_players[settings.SEASONS_COLUMNS_KEEP]
        df_players = df_players.rename(columns=settings.SEASONS_COLUMNS_RENAME)

        rows = (df_players['position_name'] == 'GK')
        column: str = 'position_name'
        df_players.loc[rows, column] = 'GKP'

        df_teams = past_data['teams'][s].copy()
        df_teams = df_teams[settings.SEASONS_TEAMS_COLUMNS_KEEP]
        df_teams = df_teams.rename(columns=settings.SEASONS_TEAMS_COLUMNS_RENAME)

        # Merge by team name (no team_id in df_players)
        l_before = len(df_players)
        df_players = df_players.merge(df_teams, on='team', how="inner")
        l_after = len(df_players)
        if l_before != l_after:
            print('Warning: Rows lost during past seasons team merge!')

        df_players = df_players.drop(columns=['team'])
        df_players = df_players.sort_values(by=['GW', 'team_name', 'position_name'], ascending=[True, True, True])

        cleaner = data_cleaner.DataCleaner(df=df_players)
        cleaner.clean_data()

        df_negative: pd.DataFrame = cleaner.flag_negative_values(
            excluded_keywords=settings.EXCLUDED_FROM_NEGATIVE_CHECK
            )
        if not df_negative.empty:
            raise ValueError("DataFrame contains incorrect negative values!")

        players_past_data[s] = df_players.copy()

    return players_past_data

def add_features(
    df_players: pd.DataFrame, players_past_data: dict[str, pd.DataFrame],
    n_form: int = 1, store: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add live and past features.
    Args:
        df_players: The pd.DataFrame of players live data.
        players_past_data: The dictionary of pd.DataFrames of players past data.
        n_form (optional): The size of the past rolling window. Defaults to 1.
        store (optional): Boolean indicating whether to store data or not. Defaults to True.
    Returns:
        A tuple of pd.DataFrames of live and past features."""

    df_features = add_features_live(df_players=df_players)

    features_past = add_features_past(players_past_data=players_past_data, n_form=n_form)
    df_features_past = merge_past_seasons(features_past=features_past)
    df_features_past = prepare_features_past(df_features_past=df_features_past)

    if store:
        file_path = f"fpl_data/clean/players_data_features.csv"
        data_store.save_df_to_csv(df=df_features, file_path=file_path)

        for s in features_past.keys():
            file_path = f"fpl_data/past/clean/season_{s}_features.csv"
            data_store.save_df_to_csv(df=features_past[s], file_path=file_path)

        file_path = f"fpl_data/past/clean/past_seasons_features_merged.csv"
        data_store.save_df_to_csv(df=df_features_past, file_path=file_path)
        # print(df.info())

    return df_features, df_features_past

def add_features_live(df_players: pd.DataFrame) -> pd.DataFrame:
    """Add live features.
    Args:
        df_players: The pd.DataFrame of players live data.
    Returns:
        A pd.DataFrame of live features."""

    fpl_features = data_features.DataFeatures(df=df_players)

    # Calculate per 90
    features_form = settings.FEATURES_FORM.copy()
    if 'DC' not in features_form:
        features_form.append('DC')
    features_form.remove('MIN')
    features_form = [c+'_form' for c in features_form]
    fpl_features.calculate_per_x(columns=features_form, x_col='MIN_form', x=90.0)

    bounds_dict = {
        'YC_form_per_90': [0.0, 2.0], 'RC_form_per_90': [0.0, 1.0],
        'BP_form_per_90': [0.0, 3.0], 'CS_form_per_90': [0.0, 1.0]
        }
    fpl_features.set_features_bounds(bounds_dict)

    df_features = fpl_features.df.copy()

    if (('OFF_home_opponent' not in df_features.columns) or
        ('OFF_away_opponent' not in df_features.columns)):

        df_features['OFF_home_opponent'], df_features['OFF_away_opponent'] = (
            set_opponent_strength(df_features=df_features, metric='xG_form'))

    if (('DEF_home_opponent' not in df_features.columns) or
        ('DEF_away_opponent' not in df_features.columns)):

        df_features['DEF_home_opponent'], df_features['DEF_away_opponent'] = (
            set_opponent_strength(df_features=df_features, metric='xGC_form'))

        df_features['DEF_home_opponent'], df_features['DEF_away_opponent'] = (
            1.0/df_features['DEF_home_opponent'], 1.0/df_features['DEF_away_opponent'])

    df_features['FDR_off'], df_features['FDR_def'] = set_fixture_difficulty(df_features)
    df_features['FDR_off'] = df_features['FDR_off'].fillna(1.0)
    df_features['FDR_def'] = df_features['FDR_def'].fillna(1.0)
    df_features['FDR_off'] = df_features['FDR_off'].clip(lower=0.0, upper=2.0)
    df_features['FDR_def'] = df_features['FDR_def'].clip(lower=0.0, upper=2.0)

    # print(df_features[columns].isna().sum())

    return df_features.copy()

def set_opponent_strength(df_features: pd.DataFrame, metric: str) -> tuple[pd.Series[float], pd.Series[float]]:
    """Set opponent strength based on opponent's specified metric.
    Args:
        df_features: The pd.DataFrame of live features.
        metric: The metric upon which strength is measured.
    Returns:
        A tuple of pd.Series[float] for the opponent's strength."""

    df_home = df_features[df_features['is_home']]
    sum_min_h = df_home.groupby('team_name')['MIN_form'].sum().replace(0, np.nan)
    sum_strength_h = df_home.groupby('team_name')[metric].sum()

    df_away = df_features[~df_features['is_home']]
    sum_min_a = df_away.groupby('team_name')['MIN_form'].sum().replace(0, np.nan)
    sum_strength_a = df_away.groupby('team_name')[metric].sum()

    #divide xGC by active players
    if 'xGC' in metric:
        count_played_h = (df_home[df_home['MIN_form'] > 0]
                          .groupby('team_name')['MIN_form'].count())
        count_played_a = (df_away[df_away['MIN_form'] > 0]
                          .groupby('team_name')['MIN_form'].count())
        sum_strength_h /= count_played_h
        sum_strength_a /= count_played_a

    # print(metric, sum_min_a['COV'], sum_strength_a['COV'])
    # print(metric, sum_min_h['COV'], sum_strength_h['COV'])

    strength_h_series = 90*(sum_strength_h/sum_min_h)
    strength_a_series = 90*(sum_strength_a/sum_min_a)

    strength_h = df_features['team_name_opponent'].map(strength_h_series)
    strength_a = df_features['team_name_opponent'].map(strength_a_series)

    return strength_h, strength_a

def set_fixture_difficulty(df_features: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """Set fixture difficulty (offensive and defensive) based on the opponent's strength.
    Args:
        df_features: The pd.DataFrame of live features.
    Returns:
        A tuple of NDArrays of the offensive and defensive fixture difficulty rating."""

    def_away_mean = df_features['DEF_away_opponent'].mean()
    def_home_mean = df_features['DEF_home_opponent'].mean()
    off_away_mean = df_features['OFF_away_opponent'].mean()
    off_home_mean = df_features['OFF_home_opponent'].mean()

    fdr_off = np.where(
        df_features['is_home'], df_features['DEF_away_opponent']/def_away_mean,
                                df_features['DEF_home_opponent']/def_home_mean
                                )

    fdr_def = np.where(
        df_features['is_home'], df_features['OFF_away_opponent']/off_away_mean,
                                df_features['OFF_home_opponent']/off_home_mean
                                )

    return fdr_off, fdr_def

def add_features_past(players_past_data: dict[str, pd.DataFrame], n_form: int = 1) -> dict[str, pd.DataFrame]:
    """Add past features.
    Args:
        players_past_data: The dictionary of pd.DataFrame of players past data.
        n_form: The size of the past rolling window.
    Returns:
        A dictionary of pd.DataFrames of past features."""

    features_form = settings.FEATURES_FORM.copy()

    past_features = {}
    for s in players_past_data.keys():

        df_players = players_past_data[s].copy()
        df_players = df_players.sort_values(['player_id', 'GW'])

        columns = ['team_id', 'team_name', 'OFF_home', 'OFF_away', 'DEF_home', 'DEF_away']
        df_opponent = df_players[columns].drop_duplicates(subset=['team_id'])
        df_players = df_players.merge(df_opponent, left_on='opponent_team_id',
            right_on='team_id', how='left', suffixes=('', '_opponent'))
        df_players = df_players.drop(columns=['opponent_team_id'])

        features_form_new = []
        for c in features_form:
            new_c = c + '_form'
            df_players[new_c] = (
                df_players.groupby('player_id')[c]
                .transform(lambda s: s.rolling(window=n_form).mean().shift(1))
                )
            # df_players = df_players.drop(columns=c)
            features_form_new.append(new_c)

        fpl_features = data_features.DataFeatures(df=df_players)
        features_form_new.remove('MIN_form')
        fpl_features.calculate_per_x(columns=features_form_new, x_col='MIN_form', x=90.0)

        bounds_dict = {
            'YC_form_per_90': [0.0, 2.0], 'RC_form_per_90': [0.0, 1.0],
            'BP_form_per_90': [0.0, 3.0], 'CS_form_per_90': [0.0, 1.0]
            }
        fpl_features.set_features_bounds(bounds_dict)

        df_players = fpl_features.df.copy()

        df_players['FDR_off'] = np.where(
            df_players['is_home'], df_players['DEF_away_opponent'],
            df_players['DEF_home_opponent'])

        df_players['FDR_def'] = np.where(
            df_players['is_home'], df_players['OFF_away_opponent'],
            df_players['OFF_home_opponent'])

        # df_players = df_players.drop(columns=[
        #     settings.FEATURES_FORM,
        #     'OFF_home', 'OFF_away', 'DEF_home', 'DEF_away', 'team_id_opponent', 'team_id',
        #     'is_home', 'OFF_home_opponent', 'OFF_away_opponent', 'DEF_home_opponent', 'DEF_away_opponent',
        #     ])

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
        'season', 'GW', 'player_name', 'position_name', 'team_name', 'PTS',
        'FDR_off', 'FDR_def'
        ]
    for f in settings.FEATURES_FORM:
        columns_seasons.append(f+'_form')

    df = df_features_past[columns_seasons].copy()
    # print(df.isna().sum())
    df = df.dropna()

    df = df.sort_values(by=['season', 'GW', 'team_name', 'player_name'], ascending=[True, True, True, True])

    return df.copy()

def set_predictions(
    df_features: pd.DataFrame, df_features_past: pd.DataFrame,
    fpl_settings: FPLSettings, store: bool = True
    ) -> pd.DataFrame:
    """Set players points predictions based on the analytical formula or a ML model.
    Args:
        df_features: The pd.DataFrame of live features.
        df_features_past: The pd.DataFrame of past features.
        fpl_settings: The FPLSettings object specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not. Defaults to True.
    Returns:
        A pd.DataFrame of live features and players points predictions."""

    if settings.PREDICTION_MODEL == "formula":
        df_predict =  set_fpl_formula_predictions(df_features=df_features, settings=fpl_settings)
    else:
        models_dict = fpl_model.set_position_models(
            df_features_past=df_features_past, test_season='2025-26'
            )
        metrics_dict = fpl_model.evaluate_position_models(
            df_features_past=df_features_past, models_dict=models_dict, test_season='2025-26'
            )
        for key, value in metrics_dict.items():
            print(key, value)

        # all data for training
        models_dict = fpl_model.set_position_models(
            df_features_past=df_features_past, test_season=None
            )
        importances = fpl_model.get_features_importances(models_dict=models_dict)
        for pos, imp in importances.items():
            print(f"\n{pos} features importance\n{imp.sort_values(ascending=False)}")

        # ML predictions
        df_predict = fpl_model.set_fpl_ml_predictions(
            df_features=df_features, models_dict=models_dict
            )
        # print(df_predict.info())

    if store:
        file_path = "fpl_data/predict/players_data_xP.csv"
        data_store.save_df_to_csv(df=df_predict, file_path=file_path)

    return df_predict

def set_fpl_formula_predictions(df_features: pd.DataFrame, fpl_settings: FPLSettings) -> pd.DataFrame:
    """Set players points predictions based on the analytical points formulas.
    Args:
        df_features: The pd.DataFrame of live features.
        fpl_settings: The FPLSettings object specifying constraints and settings.
    Returns:
        A pd.DataFrame of players points predictions."""

    df['xP_predicted'] = set_expected_points(df_features=df_features, fpl_settings=fpl_settings)

    df = df.groupby('code', as_index=False).agg(
        player_name=('player_name', 'first'), position_name=('position_name', 'first'),
        team_name=('team_name', 'first'),
        now_cost=('now_cost', 'first'), xP_predicted_sum=('xP_predicted', 'sum'),
        # availability=('availability', 'first'),
        # xP_FPL_sum=('xP_FPL', 'sum'),
        )
    df = df.sort_values(by=['team_name', 'position_name'], ascending=[True, True])

    return df.copy()

def set_expected_points(df_features: pd.DataFrame, fpl_settings: FPLSettings) -> pd.Series[float]:
    """Set the expected players points based on expected values of the related quantities.
    Args:
        df_features: The pd.DataFrame of live features.
        fpl_settings: The FPLSettings object specifying constraints and settings.
    Returns:
        A pd.Series of the expected players points."""

    fdr_off = df_features['FDR_off']
    fdr_def = df_features['FDR_def']

    xMIN = df_features['MIN_form'] # minutes

    xA = df_features['xA_form']/fdr_off # assists
    xPS = df_features['PS_form'] # penalties saved
    xBP = df_features['BP_form'] # bonus points
    xYC = df_features['YC_form'] # yellow cards
    xRC = df_features['RC_form'] # red cards
    xOG = df_features['OG_form'] # own goals
    xPM = df_features['PM_form'] # penalties missed

    cxA = fpl_settings.xA_coefficient
    cxPS = fpl_settings.xPS_coefficient
    cxYC = fpl_settings.xYC_coefficient
    cxYRC = fpl_settings.xRC_coefficient
    cxOG = fpl_settings.xOG_coefficient
    cxPM = fpl_settings.xPM_coefficient

    cxMIN_low = fpl_settings.xMIN_low_coefficient
    cxMIN_high = fpl_settings.xMIN_high_coefficient
    cxMIN_barrier = fpl_settings.xMIN_barrier
    cxMIN_threshold = fpl_settings.xMIN_threshold
    cxMIN_penalty = fpl_settings.xMIN_penalty

    cxSA = fpl_settings.xSA_coefficient
    cxGC = fpl_settings.xGC_coefficient

    cxCS = fpl_settings.xCS_coefficient
    cxCS_mid = fpl_settings.xCS_coefficient_mid

    cxDC = fpl_settings.xDC_coefficient
    cxDC_barrier_def = fpl_settings.xDC_barrier_def
    cxDC_barrier_mid = fpl_settings.xDC_barrier_mid

    cxG_gkp = fpl_settings.xG_coefficient_gkp
    cxG_def = fpl_settings.xG_coefficient_def
    cxG_mid = fpl_settings.xG_coefficient_mid
    cxG_fwd = fpl_settings.xG_coefficient_fwd

    xC = cxA*xA + cxPS*xPS + xBP - cxYC*xYC - cxYRC*xRC - cxOG*xOG - cxPM*xPM # common

    xMIN_less_than_60 = xMIN < cxMIN_barrier
    xC[xMIN_less_than_60] += cxMIN_low
    min_minutes_threshold = cxMIN_threshold
    xMIN_0 = xMIN < min_minutes_threshold
    xC[xMIN_0] -= cxMIN_penalty
    xMIN_more_than_60 = xMIN >= cxMIN_barrier
    xC[xMIN_more_than_60] += cxMIN_high

    xG = df_features['xG_form']/fdr_off
    xGC = df_features['xGC_form']*fdr_def # goals conceded
    xSA = df_features['SA_form']*fdr_def # saves
    xCS = df_features['CS_form']/fdr_def # clean sheets
    xDC = df_features['DC_form']*fdr_def # defensive contributions

    xP = pd.Series(index=df_features.index, dtype=float)

    gkp_mask = df_features["position_name"] == "GKP"
    xP[gkp_mask] = cxG_gkp*xG - cxGC*xGC + cxSA*xSA + cxCS*xCS + xC

    def_mask = df_features["position_name"] == "DEF"
    xP[def_mask] = cxG_def*xG - cxGC*xGC + cxCS*xCS + xC
    xDC10 = xDC >= cxDC_barrier_def
    xP[def_mask & xDC10] += cxDC

    mid_mask = df_features["position_name"] == "MID"
    xP[mid_mask] = cxG_mid*xG + cxCS_mid*xCS + xC
    xDC12 = xDC >= cxDC_barrier_mid
    xP[mid_mask & xDC12] += cxDC

    fwd_mask = df_features["position_name"] == "FWD"
    xP[fwd_mask] = cxG_fwd*xG + xC
    xP[fwd_mask & xDC12] += cxDC

    return xP
