
import pandas as pd
import numpy as np

from config import settings
from src.data import data_fetch, data_store, data_cleaner
from src.data.data_features import DataFeatures


def fetch_seasons_data():

    # expected values from 2022-23, DC from 2025-26
    past_seasons: list[str] = ['2022-23', '2023-24', '2024-25', '2025-26']
    file_paths: list[str] = []
    file_paths_teams: list[str] = []
    for s in past_seasons:
        file_paths.append(f"fpl_data/past/raw/season_{s}.csv")
        file_paths_teams.append(f"fpl_data/past/raw/season_{s}_teams.csv")

    prefix: str = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/'
    suffix: str = '/gws/merged_gw.csv'
    seasons_data: dict[str, pd.DataFrame] = data_fetch.fetch_multiple_csv_data(
        prefix=prefix, suffix=suffix, identifier=past_seasons, file_paths=file_paths
    )

    suffix = '/teams.csv'
    seasons_teams_data: dict[str, pd.DataFrame] = data_fetch.fetch_multiple_csv_data(
        prefix=prefix, suffix=suffix, identifier=past_seasons, file_paths=file_paths_teams
    )
    return seasons_data, seasons_teams_data

def store_seasons_data(seasons_data: dict[str, pd.DataFrame],
                       seasons_teams_data: dict[str, pd.DataFrame]) -> None:

    for s, df in seasons_data.items():
        file_path = f"fpl_data/past/raw/season_{s}.csv"
        data_store.save_df_to_csv(df=df, file_path=file_path)
    for s, df in seasons_teams_data.items():
        file_path = f"fpl_data/past/raw/season_{s}_teams.csv"
        data_store.save_df_to_csv(df=df, file_path=file_path)

def clean_seasons_data(seasons_data: dict[str, pd.DataFrame],
    seasons_teams_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Perform the cleaning process for the FPL seasons data on the dictionary of raw pd.DataFrame.
    Args:
        seasons_data: The dictionary of raw pd.DataFrame.
        seasons_teams_data: The dictionary of FPL teams pd.DataFrame.
    Returns:
        The dictionary of cleaned pd.DataFrame."""

    # missing 'code' for optimization
    seasons_columns_keep: list[str] = [
        'name', 'position', 'team', 'bonus', 'clean_sheets', 'expected_assists',
        'expected_goals', 'expected_goals_conceded', 'minutes', 'own_goals', 'penalties_missed',
        'penalties_saved', 'red_cards', 'yellow_cards', 'saves', 'total_points',
        'GW', 'opponent_team', 'was_home', 'element',
        # 'defensive_contribution'
        ]
    seasons_columns_rename = {
        'name': 'player_name', 'position': 'position_name', 'element': 'player_id', 'minutes': 'MIN',
        'expected_goals': 'xG', 'expected_assists': 'xA', 'expected_goals_conceded': 'xGC',
        'clean_sheets': 'CS', 'saves': 'SA', 'yellow_cards': 'YC', 'red_cards': 'RC',
        'penalties_missed': 'PM', 'penalties_saved': 'PS', 'own_goals': 'OG', 'bonus': 'BP',
        'total_points': 'PTS_GW', 'opponent_team': 'opponent_team_id', 'was_home': 'is_home',
        # 'defensive_contribution': 'DC'
    }
    seasons_teams_columns_keep: list[str] = [
        'id', 'name', 'short_name', 'strength_attack_home', 'strength_attack_away',
        'strength_defence_home', 'strength_defence_away'
        ]
    seasons_teams_columns_rename: dict[str, str] = {
        'id': 'team_id', 'name': 'team', 'short_name': 'team_name',
        'strength_attack_home': 'OFF_home', 'strength_attack_away': 'OFF_away',
        'strength_defence_home': 'DEF_home', 'strength_defence_away': 'DEF_away'
    }

    seasons_clean: dict[str, pd.DataFrame] = {}

    for s in seasons_data.keys():

        df_season: pd.DataFrame = seasons_data[s].copy()

        df_season = df_season[seasons_columns_keep]

        rows: pd.Series[bool] = (df_season['position'] == 'GK')
        column: str = 'position'
        df_season.loc[rows, column] = 'GKP'

        df_teams: pd.DataFrame = seasons_teams_data[s].copy()
        df_teams = df_teams[seasons_teams_columns_keep]
        df_teams = df_teams.rename(columns=seasons_teams_columns_rename)

        l_before = len(df_season)
        df_season = df_season.merge(df_teams, on='team', how="inner")
        l_after = len(df_season)
        if l_before != l_after:
            print('Warning: Rows lost during past seasons team merge!')

        df_season = df_season.rename(columns=seasons_columns_rename)
        df_season = df_season.drop(columns=['team'])
        df_season = df_season.sort_values(by=['GW', 'team_name', 'player_id'], ascending=[True, True, True])

        cleaner = data_cleaner.DataCleaner(df=df_season)
        cleaner.clean_data()

        df_negative: pd.DataFrame = cleaner.flag_negative_values(excluded_keywords=settings.EXCLUDED_FROM_NEGATIVE)
        if not df_negative.empty:
            raise ValueError("DataFrame contains incorrect negative values!")

        seasons_clean[s] = df_season.copy()

        file_path = f"fpl_data/past/clean/season_{s}_clean.csv"
        data_store.save_df_to_csv(df=seasons_clean[s], file_path=file_path)
        # print(seasons_clean[s].info())

    return seasons_clean

def set_seasons_features(seasons_clean: dict[str, pd.DataFrame],
    group_key: str, n_window: int = 1) -> dict[str, pd.DataFrame]:
    """Set FPL season features for future use on predicting optimal FPL stats.
    Args:
        seasons_clean: The cleaned dictionary of pd.DataFrame containing FPL players season data.
        group_key: The pd.DataFrame groupby keyword.
        n_window: The size of the past rolling window. Defaults to 1.
    Returns:
        A dictionary of pd.DataFrame containing additional FPL features."""

    features_average: list[str] = [
        'MIN', 'CS', 'BP', 'SA', 'YC', 'RC', 'PTS_GW',
        'PM', 'PS', 'OG',
        'xG', 'xA', 'xGC', # include in form (average)
        # 'DC'
        ]

    seasons_features: dict[str, pd.DataFrame] = {}
    for s in seasons_clean.keys():

        df_season: pd.DataFrame = seasons_clean[s].copy()
        df_season = df_season.sort_values([group_key, 'GW'])

        columns = ['team_id', 'team_name', 'OFF_home', 'OFF_away', 'DEF_home', 'DEF_away']
        df_opponent = df_season[columns].drop_duplicates(subset=['team_id'])
        df_season = df_season.merge(df_opponent, left_on='opponent_team_id',
            right_on='team_id', how='left', suffixes=('', '_opponent'))
        df_season = df_season.drop(columns=['opponent_team_id'])

        columns: list[str] = []
        for c in features_average:
            new_c = c
            if 'x' not in c:
                new_c = 'x'+c
            df_season[new_c] = (
                df_season.groupby(group_key)[c]
                .transform(lambda s: s.rolling(window=n_window).mean().shift(1))
                )
            # df_season = df_season.drop(columns=c)
            columns.append(new_c)

        fpl_features = DataFeatures(df=df_season)

        # fpl_features.convert_to_expected_from_total(columns=columns, n=n_window)
        columns.remove('xMIN')
        fpl_features.convert_to_per_x(columns=columns, x_col='xMIN', x=90.0)

        bounds_dict: dict[str, list[float]] = {
            'xYC_per_90': [0.0, 2.0], 'xRC_per_90': [0.0, 1.0],
            'xBP_per_90': [0.0, 3.0], 'xCS_per_90': [0.0, 1.0]}
        fpl_features.set_features_bounds(bounds_dict)

        columns = [c+'_per_90' for c in columns]
        fpl_features.convert_from_per_x(columns=columns, x_col='xMIN', x=90.0)

        df: pd.DataFrame = fpl_features.df.copy()

        df['FDR_off'] = np.where(df['is_home'], df['DEF_away_opponent'], df['DEF_home_opponent'])
        df['FDR_def'] = np.where(df['is_home'], df['OFF_away_opponent'], df['OFF_home_opponent'])
        df = df.drop(columns=[
            'MIN', 'CS', 'BP', 'SA', 'YC', 'RC', 'PM', 'PS', 'OG',
            'OFF_home', 'OFF_away', 'DEF_home', 'DEF_away', 'team_id_opponent', 'team_id',
            'is_home', 'OFF_home_opponent', 'OFF_away_opponent', 'DEF_home_opponent', 'DEF_away_opponent',
            ])

        seasons_features[s] = df.copy()

        file_path = f"fpl_data/past/clean/season_{s}_features.csv"
        data_store.save_df_to_csv(df=seasons_features[s], file_path=file_path)
        # print(seasons_features[s].info())

    return seasons_features

def merge_seasons(seasons_features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Set the dataset for modeling using past seasons' pd.DataFrames.
    Args:
        df_seasons: The dictionary of past seasons' pd.DataFrames.
    Returns:
        A pd.DataFrame containing the unified data."""

    df_list: list[pd.DataFrame] = []
    for s, df in seasons_features.items():
        df_new = df.copy()
        df_new['season'] = s
        df_list.append(df_new)

    df_train: pd.DataFrame = pd.concat(df_list)
    return df_train.copy()

def clean_df_seasons_merged(df_seasons_merged: pd.DataFrame) -> pd.DataFrame:
    """Perform the cleaning process for the FPL past seasons pd.DataFrame.
    Args:
        df_seasons_merged: The initial merged seasons pd.DataFrame.
    Returns:
        The cleaned pd.DataFrame."""

    columns_seasons: list[str] = [
        'season', 'GW', 'player_name', 'position_name', 'team_name', 'PTS_GW',
        'FDR_off', 'FDR_def']
    features: list[str] = [
        'xMIN', 'xCS', 'xBP', 'xSA', 'xYC', 'xRC', 'xPTS_GW',
        'xPM', 'xPS', 'xOG',
        'xG', 'xA', 'xGC',
        # 'DC'
        ]
    for f in features:
        columns_seasons.append(f)

    df: pd.DataFrame = df_seasons_merged[columns_seasons].copy()
    # print(df.isna().sum())
    df = df.dropna()

    df = df.sort_values(by=['season', 'GW', 'team_name', 'player_name'], ascending=[True, True, True, True])

    return df.copy()
