
from typing import Any
import logging

import pandas as pd

from config import settings
from src.data import data_store, data_cleaner


logger = logging.getLogger(__name__)

def prepare_players_live_data(
    live_data: dict[str, Any], fpl_settings: settings.FPLSettings, store: bool = True
    ) -> pd.DataFrame:
    """Prepare players live data for analysis.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A pd.DataFrame of players live data prepared for analysis.
    Raises:
        ValueError: If the pd.DataFrame contains not-allowed negative entries."""

    df_players = pd.DataFrame(data=live_data["elements"])

    df_players = keep_necessary_players_live_data(df_players=df_players)

    df_players.rename(columns=settings.PLAYERS_COLUMNS_RENAME, inplace=True)

    df_players['position_name'] = (df_players['element_type'].map(settings.POSITION_MAP))

    # GW, opp_id, is_home
    df_players = add_fixtures_live_data(
        df_players=df_players, fixtures_data=live_data["fixtures"],
        n_fixtures=fpl_settings.n_fixtures
        )

    # team_name, team_id_opp, team_name_opp
    df_players = add_teams_live_data(df_players, teams_data=live_data["teams"])
    df_players.drop(columns=['team_id', 'opp_id', 'team_id_opp'], inplace=True)

    (df_players.sort_values(by=['GW', 'team_name', 'element_type'],
                            ascending=[True, True, True], inplace=True))
    df_players.drop(columns='element_type', inplace=True)

    # Features form
    df_players = add_gw_live_data(
        df_players=df_players, gw_data=live_data["gw_history"], n_form=fpl_settings.n_form
        )
    df_players.drop(columns='player_id', inplace=True)

    cleaner = data_cleaner.DataCleaner(df=df_players)
    cleaner.clean_data()
    cleaner.raise_negative_values(excluded_keywords=settings.EXCLUDED_FROM_NEGATIVE_CHECK)

    df_players = cleaner.df

    if store:
        file_path = settings.LIVE_PLAYERS_STORE_CLEAN
        data_store.save_df_to_csv(df=df_players, file_path=file_path, overwrite=True)

    return df_players

def keep_necessary_players_live_data(df_players: pd.DataFrame) -> pd.DataFrame:
    """Keep only the required players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
    Returns:
        A pd.DataFrame of necessary players live data."""

    is_available = ((df_players['can_select']) & (~df_players['status'].isin(['u', 'i', 's'])))

    is_active = (df_players['status'] == 'a')

    df = df_players.loc[is_available & is_active, settings.PLAYERS_COLUMNS_KEEP].copy()

    if 'chance_of_playing_next_round' in df.columns:
        df['chance_of_playing_next_round'] = 100

    return df

def add_fixtures_live_data(
    df_players: pd.DataFrame, fixtures_data: list[dict[str, Any]], n_fixtures: int
    ) -> pd.DataFrame:
    """Add the required fixtures live data to the players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
        fixtures_data: The list of dictionaries of fixtures live data.
        n_fixtures: The number of upcoming fixtures to consider.
    Returns:
        A pd.DataFrame of players live data augmented with the fixtures live data."""

    df_fixtures = pd.DataFrame(data=fixtures_data)

    is_upcoming = (~df_fixtures['finished'])

    cols = [c for c in settings.FIXTURES_COLUMNS_KEEP if c != 'finished']
    df_fixtures = df_fixtures.loc[is_upcoming, cols].rename(columns=settings.FIXTURES_COLUMNS_RENAME)

    df_home = df_fixtures.rename(columns={'team_h': 'team_id', 'team_a': 'opp_id'})
    df_home['is_home'] = True

    df_away = df_fixtures.rename(columns={'team_a': 'team_id', 'team_h': 'opp_id'})
    df_away['is_home'] = False

    df_home_away = pd.concat([df_home, df_away], ignore_index=True)

    df_home_away.sort_values(by='GW', ascending=True, inplace=True)

    df_home_away = df_home_away.groupby('team_id').head(n_fixtures)

    df = (df_players.merge(df_home_away, on='team_id', how="outer")) # GW, opp_id, is_home

    return df

def add_teams_live_data(df_players: pd.DataFrame, teams_data: list[dict[str, Any]]) -> pd.DataFrame:
    """Add the required teams live data to the players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
        teams_data: The list of dictionaries of teams live data.
    Returns:
        A pd.DataFrame of players live data augmented with the teams live data."""

    df_teams = (
        pd.DataFrame(data=teams_data, columns=settings.TEAMS_COLUMNS_KEEP)
        .rename(columns=settings.TEAMS_COLUMNS_RENAME)
    )

    df = (
        df_players
        .merge(df_teams, on='team_id', how="outer")
        .merge(df_teams, left_on="opp_id", right_on='team_id', how="outer", suffixes=("", "_opp"))
    )  # team_name, team_id_opp, team_name_opp

    return df

def add_gw_live_data(
    df_players: pd.DataFrame, gw_data: list[list[dict[str, Any]]], n_form: int
    ) -> pd.DataFrame:
    """Add the required GW live data to the players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
        gw_data: The list of lists of dictionaries of GW live data.
        n_form: The number of past GWs to consider for players' form.
    Returns:
        A pd.DataFrame of players live data augmented with the GW live data."""

    df_players_gw = pd.concat([pd.DataFrame(data=player) for player in gw_data], ignore_index=True)

    df_players_gw = (
        df_players_gw[settings.PLAYERS_GW_COLUMNS_RENAME.keys()]
        .rename(columns=settings.PLAYERS_GW_COLUMNS_RENAME)
        .sort_values(by=['GW', 'player_id'], ascending=True)
        .groupby('player_id')
        .tail(n_form)
    )

    cleaner = data_cleaner.DataCleaner(df=df_players_gw)
    cleaner.clean_data()
    df_players_gw = cleaner.df

    features = [c.removesuffix('_form') for c in settings.FEATURES_FORM]
    if 'DC' not in features:
        features.append('DC')

    # # Keep original columns
    # df = df_players_gw
    # for c in features:
    #     df[c+'_form'] = (df_players_gw.groupby('player_id')[c].transform('mean'))

    # Drop original columns
    df_form = (
        df_players_gw.groupby('player_id')[features]
        .mean()
        .add_suffix('_form')
        .reset_index()
    )

    df = df_players.merge(df_form, on='player_id', how="left")

    return df

def prepare_players_past_data(
    past_data: dict[str, dict[str, pd.DataFrame]], store: bool = True
    ) -> dict[str, pd.DataFrame]:
    """Prepare players past data for analysis.
    Args:
        past_data: The dictionary of players, teams past data.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of pd.DataFrames of players past data prepared for analysis.
    Raises:
        ValueError: If a pd.DataFrame contains not-allowed negative entries."""

    players_past_data: dict[str, pd.DataFrame] = {}

    for s in past_data['players'].keys():

        df_players = past_data['players'][s]

        df_players = df_players[settings.SEASONS_COLUMNS_KEEP].copy()
        df_players.rename(columns=settings.SEASONS_COLUMNS_RENAME, inplace=True)

        rows = (df_players['position_name'] == 'GK')
        column = 'position_name'
        df_players.loc[rows, column] = 'GKP'

        df_teams = past_data['teams'][s]
        df_teams = df_teams[settings.SEASONS_TEAMS_COLUMNS_KEEP].copy()
        df_teams.rename(columns=settings.SEASONS_TEAMS_COLUMNS_RENAME, inplace=True)

        # Merge by team name (no team_id in df_players)
        l_before = len(df_players)
        df_players = df_players.merge(df_teams, on='team', how="inner") # team_id, team_name
        l_after = len(df_players)
        if l_before != l_after:
            logger.warning(f'Rows lost during past seasons team merge! {l_before-l_after}')

        df_players.drop(columns=['team'], inplace=True)
        df_players = df_players.sort_values(by=['GW', 'team_name', 'position_name'], ascending=[True, True, True])

        # Add opponent
        df_teams.drop(columns=['team'], inplace=True)
        df_opponent = df_teams.drop_duplicates(subset=['team_id'])
        df_players = df_players.merge(df_opponent, left_on='opp_id',
            right_on='team_id', how='left', suffixes=('', '_opp')) # team_id_opp, team_name_opp
        df_players.drop(columns=['team_id', 'opp_id', 'team_id_opp'], inplace=True)

        cleaner = data_cleaner.DataCleaner(df=df_players)
        cleaner.clean_data()
        cleaner.raise_negative_values(excluded_keywords=settings.EXCLUDED_FROM_NEGATIVE_CHECK)

        players_past_data[s] = cleaner.df

    if store:
        for s in players_past_data.keys():
            file_path = settings.PAST_STORE_CLEAN.format(season=s)
            data_store.save_df_to_csv(df=players_past_data[s], file_path=file_path, overwrite=True)

    return players_past_data
